"""Smoke test for Phase 2 temporal pipeline.

Tests individual components without going through Compose (registry scope issue).
The full pipeline is validated by actually running the training.
"""
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import mmseg
import mmseg.datasets.transforms
import custom_datasets.pastis
import custom_datasets.pastis_temporal


def test_transform():
    """Test LoadPASTISRasterTemporal transform directly (no Compose)."""
    print("=" * 60)
    print("TEST 1: LoadPASTISRasterTemporal transform")

    import json
    from custom_datasets.pastis_temporal import LoadPASTISRasterTemporal

    t = LoadPASTISRasterTemporal(img_size=256, num_classes=19)

    with open("/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R/metadata.geojson") as f:
        geojson = json.load(f)

    feat = geojson["features"][0]
    pid = int(feat["properties"]["ID_PATCH"])
    dates_raw = feat["properties"]["dates-S2"]
    dates_list = [int(v) for v in (dates_raw.values() if isinstance(dates_raw, dict) else dates_raw)]
    data_root = Path("/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R")

    results = {
        "s2_path": str(data_root / "DATA_S2" / f"S2_{pid}.npy"),
        "s2m_path": str(data_root / "DATA_S2_M12" / f"S2M_{pid}.npy"),
        "ann_path": str(data_root / "ANNOTATIONS" / f"TARGET_{pid}.npy"),
        "dates_s2": dates_list,
    }

    results = t.transform(results)
    img = results["img"]
    gt  = results["gt_seg_map"]

    print(f"  img shape: {img.shape}  (expect 256,256,120)")
    print(f"  gt shape:  {gt.shape}   (expect 256,256)")
    print(f"  img range: [{img.min():.2f}, {img.max():.2f}]")
    print(f"  gt unique: {np.unique(gt)}")
    assert img.shape == (256, 256, 120), f"Bad img shape: {img.shape}"
    print("  PASSED")
    print()


def test_backbone_fp32():
    """Test DINOv3TemporalBackbone fp32 forward."""
    print("=" * 60)
    print("TEST 2: DINOv3TemporalBackbone fp32 forward")

    from custom_models.dinov3_temporal_backbone import DINOv3TemporalBackbone
    ckpt = "/mnt/ht2_nas2/00-model/00-fb/mmseg_data/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

    backbone = DINOv3TemporalBackbone(
        arch="vit_large", patch_size=16, checkpoint=ckpt,
        interaction_indexes=[5, 11, 17, 23],
        freeze_backbone=False, in_bands=10, n_frames=12,
    ).cuda()

    print(f"  embed_dim: {backbone.embed_dim}")
    x = torch.randn(1, 120, 256, 256).cuda()
    with torch.no_grad():
        feats = backbone(x)

    for i, f in enumerate(feats):
        assert f.shape[0] == 1 and f.shape[1] == 1024, f"Bad shape: {f.shape}"
        print(f"  scale {i}: {tuple(f.shape)}")

    print("  PASSED")
    del backbone
    torch.cuda.empty_cache()
    print()


def test_backbone_bf16():
    """Test bf16 autocast forward (MSDA fallback, no NaN)."""
    print("=" * 60)
    print("TEST 3: bf16 autocast forward (no NaN)")

    from custom_models.dinov3_temporal_backbone import DINOv3TemporalBackbone
    ckpt = "/mnt/ht2_nas2/00-model/00-fb/mmseg_data/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

    backbone = DINOv3TemporalBackbone(
        arch="vit_large", patch_size=16, checkpoint=ckpt,
        interaction_indexes=[5, 11, 17, 23],
        freeze_backbone=False, in_bands=10, n_frames=12,
    ).cuda()

    x = torch.randn(1, 120, 256, 256).cuda()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            feats = backbone(x)

    for i, f in enumerate(feats):
        is_nan = torch.isnan(f).any().item()
        print(f"  scale {i}: {tuple(f.shape)}, NaN={is_nan}")
        assert not is_nan, f"NaN in scale {i}"

    print("  PASSED")
    del backbone
    torch.cuda.empty_cache()
    print()


def test_bs4_forward():
    """Test with batch_size=4 to check memory."""
    print("=" * 60)
    print("TEST 4: bs=4 forward (memory check)")

    from custom_models.dinov3_temporal_backbone import DINOv3TemporalBackbone
    ckpt = "/mnt/ht2_nas2/00-model/00-fb/mmseg_data/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

    backbone = DINOv3TemporalBackbone(
        arch="vit_large", patch_size=16, checkpoint=ckpt,
        interaction_indexes=[5, 11, 17, 23],
        freeze_backbone=False, in_bands=10, n_frames=12,
    ).cuda()

    x = torch.randn(4, 120, 256, 256).cuda()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            feats = backbone(x)

    alloc_mb = torch.cuda.max_memory_allocated() / 1e6
    for i, f in enumerate(feats):
        print(f"  scale {i}: {tuple(f.shape)}")

    print(f"  Peak GPU memory: {alloc_mb:.1f} MB")
    print("  PASSED" if alloc_mb < 44000 else "  HIGH MEMORY WARNING")
    del backbone
    torch.cuda.empty_cache()
    print()


if __name__ == "__main__":
    test_transform()
    print("Loading ViT-L model (~3 GB, ~2 min)...\n")
    test_backbone_fp32()
    test_backbone_bf16()
    test_bs4_forward()
    print("All smoke tests passed.")
