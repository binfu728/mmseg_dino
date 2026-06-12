"""Offline pre-compute monthly median composites for PASTIS-R.

Reads S2_{pid}.npy (T, 10, 128, 128), computes 12-month median composite,
writes S2M_{pid}.npy (12, 10, 128, 128) as float16 to save disk space.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PASTIS_ROOT = Path("/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R")
OUT_DIR = PASTIS_ROOT / "DATA_S2_M12"


def compute_monthly_median(s2_path, dates_list):
    s2 = np.load(s2_path).astype(np.float32)
    months = np.array([int(str(d)[4:6]) for d in dates_list], dtype=np.int32)
    frames = []
    for m in range(1, 13):
        sel = s2[months == m]
        if len(sel) == 0:
            indices = np.abs(months - m).argsort()[:3]
            sel = s2[indices]
        frames.append(np.median(sel, axis=0))
    return np.stack(frames).astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    with open(PASTIS_ROOT / "metadata.geojson") as f:
        geojson = json.load(f)

    total = len(geojson["features"])
    done = skipped = 0

    for feat in geojson["features"]:
        pid = int(feat["properties"]["ID_PATCH"])
        s2_path = PASTIS_ROOT / "DATA_S2" / f"S2_{pid}.npy"
        out_path = OUT_DIR / f"S2M_{pid}.npy"

        if not s2_path.exists():
            continue
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        dates_raw = feat["properties"]["dates-S2"]
        dates_list = [int(v) for v in (dates_raw.values() if isinstance(dates_raw, dict) else dates_raw)]

        monthly = compute_monthly_median(s2_path, dates_list)
        np.save(out_path, monthly)
        done += 1

        if (done + skipped) % 200 == 0:
            print(f"Progress: {done + skipped}/{total} (new={done}, skipped={skipped})")

    print(f"Done. Created {done} files, skipped {skipped} in {OUT_DIR}")


if __name__ == "__main__":
    main()
