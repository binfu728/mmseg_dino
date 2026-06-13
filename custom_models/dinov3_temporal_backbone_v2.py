"""DINOv3 temporal backbone wrapper: per-frame encoding + temporal mean pooling.

Input (B, T*C, H, W) is split into T frames, each frame goes through
ViT+Adapter independently, then features are mean-pooled across time at each 
of the 4 output scales.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

from mmseg.registry import MODELS

_DINO_ROOT = str(Path(__file__).parents[5] / "dino" / "dinov3")
if _DINO_ROOT not in sys.path:
    sys.path.insert(0, _DINO_ROOT)

# Sentinel-2 B04(R), B03(G), B02(B)在10波段中的索引
_BAND_OF_RGB = [2, 1, 0]


def _inflate_conv(conv: nn.Conv2d, in_bands: int) -> nn.Conv2d:
    """将原本的3通道Conv膨胀至指定的in_bands通道，初始化RGB相关权重并均分剩余波段"""
    new = nn.Conv2d(
        in_bands, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        w = conv.weight  # (out, 3, kh, kw)
        # 将RGB权重的均值按比例复制给所有通道
        new.weight.copy_(w.mean(dim=1, keepdim=True).repeat(1, in_bands, 1, 1) * (3.0 / in_bands))
        # 特别地，把原本的RGB预训练权重精准放入对应的遥感波段通道中
        for rgb_idx, band_idx in enumerate(_BAND_OF_RGB):
            new.weight[:, band_idx] = w[:, rgb_idx]
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    new.weight.requires_grad_(True)
    if new.bias is not None:
        new.bias.requires_grad_(True)
    return new


class _MmcvMSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        super().__init__()
        self.attn = MultiScaleDeformableAttention(
            embed_dims=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )

    def init_weights(self):
        self.attn.init_weights()

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        return self.attn(
            query=query,
            value=input_flatten,
            identity=torch.zeros_like(query),
            query_pos=None,
            key_padding_mask=input_padding_mask,
            reference_points=reference_points,
            spatial_shapes=input_spatial_shapes,
            level_start_index=input_level_start_index,
        )


@MODELS.register_module()
class DINOv3TemporalBackbone_v2(BaseModule):
    _DEFAULT_INTERACTION_INDEXES = {
        "vit_small": [2, 5, 8, 11],
        "vit_base":  [2, 5, 8, 11],
        "vit_large": [5, 11, 17, 23],
        "vit_huge":  [7, 15, 23, 31],
    }

    def __init__(
        self,
        arch: str = "vit_base",
        patch_size: int = 16,
        checkpoint=None,
        interaction_indexes=None,
        freeze_backbone: bool = False,
        in_bands: int = 10,
        n_frames: int = 12,
        init_cfg=None,
    ):
        super().__init__(init_cfg=None)

        from omegaconf import OmegaConf
        from dinov3.models import build_model_for_eval
        from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter

        self.n_frames = n_frames
        self.in_bands_per_frame = in_bands

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

        vit = build_model_for_eval(cfg, pretrained_weights=checkpoint)

        if interaction_indexes is None:
            interaction_indexes = self._DEFAULT_INTERACTION_INDEXES.get(
                arch, [2, 5, 8, 11]
            )

        self.adapter = DINOv3_Adapter(vit, interaction_indexes=interaction_indexes)
        self._replace_msda_with_mmcv()

        # 修复：执行 Stem 权重通道膨胀以接受10通道遥感影像
        if in_bands != 3:
            vit_bb = self.adapter.backbone
            vit_bb.patch_embed.proj = _inflate_conv(vit_bb.patch_embed.proj, in_bands)
            vit_bb.patch_embed.in_chans = in_bands
            self.adapter.spm.stem[0] = _inflate_conv(self.adapter.spm.stem[0], in_bands)

        if not freeze_backbone:
            self.adapter.backbone.requires_grad_(True)

        self.embed_dim = vit.embed_dim

    def _replace_msda_with_mmcv(self):
        from dinov3.eval.segmentation.models.utils.ms_deform_attn import MSDeformAttn

        for parent in self.adapter.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, MSDeformAttn):
                    replacement = _MmcvMSDeformAttn(
                        d_model=child.d_model,
                        n_levels=child.n_levels,
                        n_heads=child.n_heads,
                        n_points=child.n_points,
                        ratio=child.ratio,
                    )
                    replacement.init_weights()
                    setattr(parent, name, replacement)

    def forward(self, x):
        B, TC, H, W = x.shape
        T = self.n_frames
        C = self.in_bands_per_frame
        assert TC == T * C, f"Expected {T}*{C} channels, got {TC}"
        
        # 折叠时间维度到Batch维度
        x = x.view(B * T, C, H, W)
        
        # 逐帧送入Adapter
        out = self.adapter(x)

        # 聚合特征(时序平均)
        feats = []
        for key in ["1", "2", "3", "4"]:
            f = out[key]
            BT, D, h, w = f.shape
            f = f.view(B, T, D, h, w)
            f = f.mean(dim=1)
            feats.append(f)
        return tuple(feats)