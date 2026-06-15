"""DINOv3 ViT backbone wrapper for mmsegmentation."""
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

from mmseg.registry import MODELS

_DINO_ROOT = str(Path(__file__).parents[5] / "dino" / "dinov3")
if _DINO_ROOT not in sys.path:
    sys.path.insert(0, _DINO_ROOT)


class _MmcvMSDeformAttn(nn.Module):
    """Drop-in replacement for DINOv3's MSDeformAttn backed by mmcv's CUDA extension."""

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
class DINOv3BackboneMmseg(BaseModule):
    """DINOv3 ViT + DINOv3_Adapter wrapped as an mmseg backbone.

    Returns a tuple of 4 feature maps at strides [4, 8, 16, 32].
    All outputs have `embed_dim` channels (1024 for ViT-L).

    Args:
        arch: ViT variant, e.g. 'vit_large'.
        patch_size: ViT patch size (16).
        checkpoint: Path to pretrained weights (.pth file or DCP directory).
        interaction_indexes: Which transformer block outputs to use for the
            4 interaction stages.
        freeze_backbone: Whether to keep ViT weights frozen.
        in_bands: Number of input bands. When != 3, a learnable 1x1 Conv
            projects to 3-channel RGB before the ViT, initialized with
            RGB bands copied through and remaining bands set to zero.
        init_cfg: Ignored; weight loading is handled by build_model_for_eval.
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

        if not freeze_backbone:
            self.adapter.backbone.requires_grad_(True)

        self.embed_dim = vit.embed_dim

        if in_bands != 3:
            self.input_proj = nn.Conv2d(in_bands, 3, kernel_size=1)
            nn.init.zeros_(self.input_proj.weight)
            for rgb_i, band_i in enumerate([2, 1, 0]):
                self.input_proj.weight.data[rgb_i, band_i, 0, 0] = 1.0
            nn.init.zeros_(self.input_proj.bias)
        else:
            self.input_proj = nn.Identity()

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
        x = self.input_proj(x)
        out = self.adapter(x)
        return (out["1"], out["2"], out["3"], out["4"])
