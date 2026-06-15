"""DINOv3 ViT backbone wrapper for mmsegmentation (V2).

Compared to V1 (dinov3_backbone.py):
  - Replaces 1×1 input_proj (Conv 10→3) with stem inflation:
    directly inflates ViT patch_embed.proj and Adapter spm.stem[0]
    from 3 channels to in_bands channels.
  - No temporal logic (single-frame, same as V1 forward).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

from mmseg.registry import MODELS

_DINO_ROOT = '/mnt/ht2-nas2/00-model/00-fb/mmseg_dino/dinov3'
if _DINO_ROOT not in sys.path:
    sys.path.insert(0, _DINO_ROOT)

_BAND_OF_RGB = [2, 1, 0]


def _inflate_conv(conv: nn.Conv2d, in_bands: int) -> nn.Conv2d:
    new = nn.Conv2d(
        in_bands, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        w = conv.weight
        new.weight.copy_(w.mean(dim=1, keepdim=True).repeat(1, in_bands, 1, 1) * (3.0 / in_bands))
        for rgb_idx, band_idx in enumerate(_BAND_OF_RGB):
            new.weight[:, band_idx] = w[:, rgb_idx]
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
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
class DINOv3BackboneMmseg_v2(BaseModule):
    """DINOv3 ViT + DINOv3_Adapter wrapped as an mmseg backbone (V2).

    V2 improvement: stem inflation instead of 1×1 input_proj.
    When in_bands != 3, ViT.patch_embed.proj and Adapter.spm.stem[0]
    are inflated from 3→in_bands channels with RGB weights copied.

    Returns a tuple of 4 feature maps at strides [4, 8, 16, 32].
    All outputs have `embed_dim` channels (1024 for ViT-L).

    Args:
        arch: ViT variant, e.g. 'vit_large'.
        patch_size: ViT patch size (16).
        checkpoint: Path to pretrained weights.
        interaction_indexes: Which transformer block outputs to use for the
            4 interaction stages.
        freeze_backbone: Whether to keep ViT weights frozen.
        in_bands: Number of input bands (3=RGB; 10=full Sentinel-2).
            When in_bands != 3, stem layers are inflated to accept in_bands
            channels, with pre-trained RGB weights copied through.
        init_cfg: Ignored.
    """

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
        in_bands: int = 3,
        init_cfg=None,
    ):
        super().__init__(init_cfg=None)

        from omegaconf import OmegaConf
        from dinov3.models import build_model_for_eval
        from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter

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
        out = self.adapter(x)
        return (out["1"], out["2"], out["3"], out["4"])
