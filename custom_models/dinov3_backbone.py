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
    """Drop-in replacement for DINOv3's MSDeformAttn backed by mmcv's CUDA extension.

    Matches DINOv3's MSDeformAttn __init__ and forward signatures exactly so it
    can be swapped in-place inside DINOv3_Adapter without touching adapter code.

    `ratio` is accepted for API compatibility but not used — mmcv's MSDA always
    operates at full d_model. This differs from DINOv3's default ratio=0.5 but
    avoids a double value-projection bug. Requires starting training fresh (no
    --resume from checkpoints saved with the original MSDeformAttn).

    mmcv adds an internal residual; we suppress it (identity=zeros) because
    Extractor.forward() adds the residual externally.
    """

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
    All outputs have `embed_dim` channels (768 for ViT-B, 1024 for ViT-L).

    Args:
        arch: ViT variant, e.g. 'vit_base' or 'vit_large'.
        patch_size: ViT patch size (16 or 14).
        checkpoint: Path to pretrained weights (.pth file or DCP directory).
            Pass None or '' to train from scratch.
        interaction_indexes: Which transformer block outputs to use for the
            4 interaction stages. Defaults suit ViT-B (12 blocks).
        freeze_backbone: Whether to keep ViT weights frozen. The adapter
            interaction layers are always trainable.
        init_cfg: Ignored; weight loading is handled by build_model_for_eval.
    """

    # ViT-B has 12 blocks; use four evenly spaced indices.
    _DEFAULT_INTERACTION_INDEXES = {
        "vit_small": [2, 5, 8, 11],
        "vit_base": [2, 5, 8, 11],
        "vit_large": [5, 11, 17, 23],
        "vit_huge": [7, 15, 23, 31],
    }

    def __init__(
        self,
        arch: str = "vit_base",
        patch_size: int = 16,
        checkpoint=None,
        interaction_indexes=None,
        freeze_backbone: bool = False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=None)  # skip mmengine weight init

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

        # Replace every MSDeformAttn in the adapter with the mmcv-backed wrapper.
        # This must happen after DINOv3_Adapter.__init__ (which calls _reset_parameters
        # on the original modules) so the new modules get their own init_weights call.
        self._replace_msda_with_mmcv()

        # DINOv3_Adapter always freezes the inner backbone; optionally unfreeze.
        if not freeze_backbone:
            self.adapter.backbone.requires_grad_(True)

        self.embed_dim = vit.embed_dim

    def _replace_msda_with_mmcv(self):
        """Swap all DINOv3 MSDeformAttn modules with _MmcvMSDeformAttn."""
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
        # DINOv3_Adapter returns {"1": f1, "2": f2, "3": f3, "4": f4}
        # strides: f1=4, f2=8, f3=16, f4=32
        out = self.adapter(x)
        return (out["1"], out["2"], out["3"], out["4"])
