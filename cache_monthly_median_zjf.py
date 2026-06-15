"""Precompute monthly median S2 composites for PASTIS-R.

Writes <data_root>/DATA_S2_M12/S2M_<ID>.npy as float16 (12, 10, 128, 128),
raw reflectance (normalisation happens in the loader).

Usage:
    conda run -n olmoearth python3 jzf/cache_monthly_median.py [--workers 8]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from jzf.pastis_temporal import monthly_median, CACHE_DIRNAME  # noqa: E402

DATA_ROOT = Path("/home/zifei/dataset/PASTIS-R")


def _process(item):
    pid, dates = item
    out_path = DATA_ROOT / CACHE_DIRNAME / f"S2M_{pid}.npy"
    if out_path.exists():
        return pid, "skip"
    s2_path = DATA_ROOT / "DATA_S2" / f"S2_{pid}.npy"
    if not s2_path.exists():
        return pid, "missing"
    s2 = np.load(s2_path).astype(np.float32)
    frames = monthly_median(s2, dates)
    np.save(out_path, frames.astype(np.float16))
    return pid, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    (DATA_ROOT / CACHE_DIRNAME).mkdir(exist_ok=True)
    with open(DATA_ROOT / "metadata.geojson") as f:
        geojson = json.load(f)

    items = []
    for feat in geojson["features"]:
        props = feat["properties"]
        dates = props["dates-S2"]
        if isinstance(dates, dict):
            dates = [dates[k] for k in sorted(dates, key=int)]
        items.append((int(props["ID_PATCH"]), dates))

    counts = {"ok": 0, "skip": 0, "missing": 0}
    with Pool(args.workers) as pool:
        for i, (pid, status) in enumerate(pool.imap_unordered(_process, items, chunksize=8)):
            counts[status] += 1
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(items)}  {counts}", flush=True)
    print("done:", counts)


if __name__ == "__main__":
    main()
