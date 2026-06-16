"""DINOv3 temporal backbone wrapper: per-frame encoding + temporal attention pooling.

Input (B, T*C, H, W) is split into T frames, each frame goes through
ViT+Adapter independently, then features are pooled across time at each 
of the 4 output scales (AttnPool for scales 2/3/4, MeanPool for scale 1).
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


class TemporalAttnPool(nn.Module):
    """时序注意力池化层 (Temporal Attention Pooling)
    
    聚合 (B, T, D, h, w) 维度数据为 (B, D, h, w)。
    使用一个可学习的 Query 来对 T 个时相帧进行注意力加权，并结合了月份位置编码。
    """
    def __init__(self, dim: int, n_frames: int, n_heads: int = 8):
        super().__init__()
        self.month_embed = nn.Parameter(torch.zeros(n_frames, dim))
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.month_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x):
        B, T, D, h, w = x.shape
        # 将空间维度拍平合并进 Batch 维度，变成 (B*h*w, T, D) 以便对每个像素进行时序注意力计算
        x = x.permute(0, 3, 4, 1, 2).reshape(B * h * w, T, D)
        # 注入月份位置编码并归一化
        x = self.norm(x + self.month_embed)
        # 将 Query 扩展到所有像素
        q = self.query.expand(B * h * w, -1, -1)
        # 计算注意力加权求和，输出 (B*h*w, 1, D)
        out, _ = self.attn(q, x, x)
        # 还原回空间形状 (B, D, h, w)
        return out.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()


@MODELS.register_module()
class DINOv3TemporalBackbone_v3(BaseModule):
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
        drop_path_rate: float = 0,
        temporal_agg: str = "attn",  # 新增: 聚合方式 "mean", "max" 或 "attn"
        attn_heads: int = 8,         # 新增: 注意力头数
        init_cfg=None,
    ):
        super().__init__(init_cfg=None)

        from omegaconf import OmegaConf
        from dinov3.models import build_model_for_eval
        from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter

        self.n_frames = n_frames
        self.in_bands_per_frame = in_bands
        self.temporal_agg = temporal_agg

        cfg = OmegaConf.create({
            "student": {
                "arch": arch,
                "patch_size": patch_size,
                "drop_path_rate":drop_path_rate,
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

        # ---------------------------------------------------------
        # 新增: 注册时序注意力池化模块
        # 对 8/16/32 下采样尺度(key="2", "3", "4")开启注意力机制
        # stride-4(key="1")尺寸太大，为防止OOM强制降级使用 mean
        # ---------------------------------------------------------
        self.tpool = nn.ModuleDict()
        if temporal_agg == "attn":
            for key in ("2", "3", "4"):
                self.tpool[key] = TemporalAttnPool(
                    self.embed_dim, n_frames, n_heads=attn_heads)

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

    def _aggregate(self, f, key):
        """时序聚合分发器"""
        if self.temporal_agg == "attn" and key in self.tpool:
            return self.tpool[key](f)
        if self.temporal_agg == "max":
            return f.amax(dim=1)
        # 兜底：Mean Pool (用于 temporal_agg="mean" 或 stride=4 特征)
        return f.mean(dim=1)

    def forward(self, x):
        B, TC, H, W = x.shape
        T = self.n_frames
        C = self.in_bands_per_frame
        assert TC == T * C, f"Expected {T}*{C} channels, got {TC}"
        
        # 折叠时间维度到Batch维度
        x = x.view(B * T, C, H, W)
        
        # 逐帧送入Adapter
        out = self.adapter(x)

        # 聚合特征(根据配置动态选择 Attn/Mean/Max)
        feats = []
        for key in ["1", "2", "3", "4"]:
            f = out[key]
            BT, D, h, w = f.shape
            # 还原出 T 维度: (B, T, D, h, w)
            f = f.view(B, T, D, h, w)
            
            # 使用聚合方法融合 T
            f = self._aggregate(f, key)
            feats.append(f)
            
        return tuple(feats)