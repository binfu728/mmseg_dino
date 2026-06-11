"""DINOv3 纯 ViT backbone for mmsegmentation — 时间维度特征池化版本.

与 dinov3_backbone.py 的区别:
    - 不使用 DINOv3_Adapter (SPM + 4 层交互), 因为 FCN 只需要单层输出
    - 直接使用原始 DinoVisionTransformer, 在时间维度展开/池化
    - 输出 1 个特征图 (而非 4 个多尺度特征), auxiliary_head 需置为 None

用法:
    backbone=dict(
        type='DINOv3BackboneMmsegTemporal',
        arch='vit_large',
        patch_size=16,
        checkpoint='path/to/dinov3_vitl16_pretrain.pth',
        freeze_backbone=False,
    )
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmseg.registry import MODELS  # type: ignore

_DINO_ROOT = str(Path(__file__).resolve().parent.parent / "dinov3")
if _DINO_ROOT not in sys.path:
    sys.path.insert(0, _DINO_ROOT)


# ViT-Large 有 24 层, 取最后一层 (0-indexed: 23)
_LAST_BLOCK_INDEX = {
    "vit_small": 11,
    "vit_base": 11,
    "vit_large": 23,
    "vit_huge": 31,
}


@MODELS.register_module()
class DINOv3BackboneMmsegTemporal(BaseModule):
    """DINOv3 ViT backbone with temporal feature pooling for mmseg.

    输入: (B, T×3, H, W) — T 个时相的 RGB 通道拼接
    处理:
        1. 展开时间维度: (B, T×3, H, W) → (B×T, 3, H, W)
        2. 通过原始 ViT 提取最后一层 patch tokens
        3. 折叠时间维度: (B×T, C, H_f, W_f) → (B, T, C, H_f, W_f)
        4. 时间维度 Mean Pooling: (B, T, C, H_f, W_f) → (B, C, H_f, W_f)
    输出: (feat_pooled,) — 单元素 tuple, stride = patch_size

    Args:
        arch: ViT 变体 ('vit_small' | 'vit_base' | 'vit_large' | 'vit_huge').
        patch_size: patch 大小 (14 或 16).
        checkpoint: 预训练权重路径 (.pth).
        freeze_backbone: True=冻结 ViT; False=全参微调.
    """

    def __init__(
        self,
        arch: str = "vit_large",
        patch_size: int = 16,
        checkpoint=None,
        freeze_backbone: bool = False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=None)

        from omegaconf import OmegaConf
        from dinov3.models import build_model_for_eval

        cfg = OmegaConf.create({
            "student": {
                "arch": arch,
                "patch_size": patch_size,
                "pos_embed_rope_base": None,
                "pos_embed_rope_min_period": 4,
                "pos_embed_rope_max_period": 50,
                "pos_embed_rope_normalize_coords": "separate",
                "pos_embed_rope_shift_coords": None,
                "pos_embed_rope_jitter_coords": None,
                "pos_embed_rope_rescale_coords": None,
                "qkv_bias": True,
                "layerscale": 1e-5,
                "norm_layer": "layernorm",
                "ffn_layer": "mlp",
                "ffn_bias": True,
                "proj_bias": True,
                "n_storage_tokens": 0,
                "mask_k_bias": False,
                "untie_cls_and_patch_norms": False,
                "untie_global_and_local_cls_norm": False,
                "fp8_enabled": False,
            },
            "crops": {"global_crops_size": 224},
        })

        self.vit = build_model_for_eval(cfg, pretrained_weights=checkpoint)

        if not freeze_backbone:
            self.vit.requires_grad_(True)

        self.embed_dim = self.vit.embed_dim
        self.patch_size = patch_size

        self._last_block_idx = _LAST_BLOCK_INDEX.get(arch, 23)
        # 最后一层 block 的索引列表, get_intermediate_layers 需要 list
        self._last_block_indices = [self._last_block_idx]

    def forward(self, x: torch.Tensor):
        """Forward with temporal unfolding and feature pooling.

        Args:
            x: (B, T×3, H, W) 多时相 RGB 拼接, 例如 T=12 → 36 通道.

        Returns:
            tuple: (feat_pooled,) — (B, embed_dim, H/patch_size, W/patch_size).
        """
        B, C_in, H, W = x.shape
        T = C_in // 3

        # 1. 展开时间维度到 batch: (B, T×3, H, W) → (B×T, 3, H, W)
        x = x.view(B, T, 3, H, W).reshape(B * T, 3, H, W)

        # 2. 通过原始 ViT, 取最后一层 patch tokens
        #    reshape=True → 输出已转为 (B×T, embed_dim, H_f, W_f) 空间格式
        out = self.vit.get_intermediate_layers(
            x,
            n=self._last_block_indices,
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        feat = out[0]  # (B×T, embed_dim, H_f, W_f)

        # 3. 折叠时间维度 + Mean Pooling
        _, C_feat, H_f, W_f = feat.shape
        feat = feat.view(B, T, C_feat, H_f, W_f)
        feat_pooled = feat.mean(dim=1)  # (B, embed_dim, H_f, W_f)

        return (feat_pooled,)
