"""Smoke test for the temporal PASTIS pipeline: data shapes, one train_step
(AMP, bs=4), one val_step, and peak GPU memory.

Usage:
    cd <mmseg root> && PYTHONPATH=. conda run -n olmoearth python3 jzf/smoke_test.py
"""
import sys
from pathlib import Path

MMSEG_ROOT = str(Path(__file__).parents[1])
sys.path.insert(0, MMSEG_ROOT)

import torch
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.optim import build_optim_wrapper
from mmengine.registry import init_default_scope

from mmseg.registry import DATASETS, MODELS


def main():
    cfg = Config.fromfile(f"{MMSEG_ROOT}/jzf/configs/dinov3s_m2f_pastis_temporal.py")
    init_default_scope("mmseg")

    # --- dataset ---
    ds = DATASETS.build(cfg.train_dataloader.dataset)
    print(f"train samples: {len(ds)}")
    s = ds[0]
    img = s["inputs"]
    gt = s["data_samples"].gt_sem_seg.data
    print(f"inputs: {tuple(img.shape)} {img.dtype}   gt: {tuple(gt.shape)}, "
          f"labels: {sorted(torch.unique(gt).tolist())[:8]}...")
    assert img.shape[0] == 120 and img.shape[1] == 256

    # --- model ---
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)  # Runner does this when non-distributed
    model = model.cuda()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.1f}M")
    pe = model.backbone.adapter.backbone.patch_embed.proj
    print(f"patch_embed.proj: in={pe.in_channels} out={pe.out_channels}")
    spm0 = model.backbone.adapter.spm.stem[0]
    print(f"spm.stem[0]: in={spm0.in_channels} out={spm0.out_channels}")

    optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)

    batch = {
        "inputs": [ds[i]["inputs"] for i in range(4)],
        "data_samples": [ds[i]["data_samples"] for i in range(4)],
    }

    # --- train step (bs=4, AMP) ---
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        log_vars = model.train_step(batch, optim_wrapper)
        loss_str = ", ".join(f"{k}={float(v):.3f}" for k, v in list(log_vars.items())[:4])
        print(f"train step {step}: {loss_str}")
    print(f"peak GPU mem (train, bs=4): {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    # --- val step (bs=2) ---
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model.val_step({
            "inputs": batch["inputs"][:2],
            "data_samples": batch["data_samples"][:2],
        })
    print(f"pred shape: {tuple(out[0].pred_sem_seg.data.shape)}, "
          f"pred classes: {sorted(torch.unique(out[0].pred_sem_seg.data).tolist())[:8]}")
    print(f"peak GPU mem (val, bs=2): {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
