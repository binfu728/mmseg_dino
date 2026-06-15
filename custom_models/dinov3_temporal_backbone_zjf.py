"""DINOv3 temporal backbone: 10-band stems + per-frame encoding + temporal pooling.

Phase 1+2 of docs_zf/dinov3_pastis_mIoU提升方案.md:

* The ViT ``patch_embed.proj`` and the Adapter ``spm.stem[0]`` convs are
  inflated from 3 to ``in_bands`` input channels: the pretrained R/G/B
  kernels are copied into the B04/B03/B02 slots, the remaining bands start
  from the RGB-mean kernel scaled by 3/in_bands.
* The input is a stack of ``n_frames`` monthly composites along channels:
  (B, n_frames*in_bands, H, W).  Each frame goes through ViT+Adapter
  independently; the 4 multi-scale feature maps are mean-pooled over time.
"""
import torch
import torch.nn as nn
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

from mmseg.registry import MODELS
from mmseg.models.backbones.dinov3_backbone import DINOv3BackboneMmseg

# mmcv's ms_deform_attn CUDA kernel has no BFloat16 variant; under bf16
# autocast (used because fp16 NaNs in the M2F mask BCE) run MSDA in fp32.
# Patching the Function covers both the DINOv3 adapter and the M2F pixel
# decoder, which use the same mmcv op.
_ORIG_MSDA_APPLY = MultiScaleDeformableAttnFunction.apply


def _msda_apply_fp32(value, spatial_shapes, level_start_index,
                     sampling_locations, attention_weights, im2col_step):
    if value.dtype in (torch.bfloat16, torch.float16):
        out = _ORIG_MSDA_APPLY(
            value.float(), spatial_shapes, level_start_index,
            sampling_locations.float(), attention_weights.float(),
            im2col_step)
        return out.to(value.dtype)
    return _ORIG_MSDA_APPLY(value, spatial_shapes, level_start_index,
                            sampling_locations, attention_weights,
                            im2col_step)


MultiScaleDeformableAttnFunction.apply = _msda_apply_fp32

# Indices of B04 (R), B03 (G), B02 (B) within the 10-band S2 stack
_BAND_OF_RGB = [2, 1, 0]


def _inflate_conv(conv: nn.Conv2d, in_bands: int) -> nn.Conv2d:
    new = nn.Conv2d(
        in_bands, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        w = conv.weight  # (out, 3, kh, kw), RGB order
        new.weight.copy_(w.mean(dim=1, keepdim=True).repeat(1, in_bands, 1, 1)
                         * (3.0 / in_bands))
        for rgb_idx, band_idx in enumerate(_BAND_OF_RGB):
            new.weight[:, band_idx] = w[:, rgb_idx]
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    new.weight.requires_grad_(True)
    if new.bias is not None:
        new.bias.requires_grad_(True)
    return new


@MODELS.register_module()
class DINOv3TemporalBackbone(DINOv3BackboneMmseg):
    """Wraps DINOv3BackboneMmseg for multi-band, multi-frame input.

    Args (in addition to DINOv3BackboneMmseg's):
        in_bands: spectral bands per frame (10 for Sentinel-2).
        n_frames: temporal frames stacked along the channel dim.
        temporal_agg: 'mean' or 'max' pooling over the T dimension.
    """

    def __init__(self, *args, in_bands: int = 10, n_frames: int = 12,
                 temporal_agg: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        assert temporal_agg in ("mean", "max")
        self.in_bands = in_bands
        self.n_frames = n_frames
        self.temporal_agg = temporal_agg

        if in_bands != 3:
            vit = self.adapter.backbone
            vit.patch_embed.proj = _inflate_conv(vit.patch_embed.proj, in_bands)
            vit.patch_embed.in_chans = in_bands
            self.adapter.spm.stem[0] = _inflate_conv(
                self.adapter.spm.stem[0], in_bands)

    def forward(self, x):
        B, TC, H, W = x.shape
        T = self.n_frames
        assert TC == T * self.in_bands, \
            f"expected {T}*{self.in_bands} channels, got {TC}"
        x = x.view(B * T, self.in_bands, H, W)
        out = self.adapter(x)
        feats = []
        for key in ("1", "2", "3", "4"):
            f = out[key]                       # (B*T, D, h, w)
            f = f.view(B, T, *f.shape[1:])
            f = f.mean(dim=1) if self.temporal_agg == "mean" else f.amax(dim=1)
            feats.append(f)
        return tuple(feats)
