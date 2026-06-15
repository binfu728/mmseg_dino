"""PASTIS-R temporal (monthly median composite) dataset for mmsegmentation.

Phase 1+2 of docs_zf/dinov3_pastis_mIoU提升方案.md:

* 10 spectral bands (B02..B12), normalised inside the loader with the
  fold-averaged stats from NORM_S2_patch.json.
* 12 monthly median composite frames stacked along channels:
  output img is (H, W, 12*10); the backbone un-stacks them.
* U-TAE evaluation protocol: 19 classes (0 = background participates in
  mIoU, 1-18 = crops), only void (19) -> ignore_index 255.
* RandomRotate90 augmentation (remote sensing has no orientation prior).

Monthly composites are read from <data_root>/DATA_S2_M12/S2M_<ID>.npy
(float16, (12, 10, 128, 128)) when present — produce them with
jzf/cache_monthly_median.py — otherwise computed on the fly.
"""
from pathlib import Path
import json

import cv2
import numpy as np

from mmcv.transforms import BaseTransform
from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.datasets.basesegdataset import BaseSegDataset

# Band order in S2_*.npy: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
# Fold-averaged stats from NORM_S2_patch.json (see jzf/cache_monthly_median.py)
PASTIS_S2_MEAN = np.array(
    [1180.2, 1387.7, 1436.7, 1773.6, 2735.8, 3080.1, 3223.6, 3338.3, 2418.1, 1630.2],
    dtype=np.float32)
PASTIS_S2_STD = np.array(
    [1976.7, 1916.8, 1996.2, 1903.1, 1784.9, 1796.3, 1811.8, 1793.3, 1474.4, 1309.8],
    dtype=np.float32)

_SPLIT_FOLDS = {"train": [1, 2, 3], "val": [4], "test": [5]}

CACHE_DIRNAME = "DATA_S2_M12"


def monthly_median(s2: np.ndarray, dates, n_frames: int = 12) -> np.ndarray:
    """(T, 10, H, W) raw S2 stack + acquisition dates (YYYYMMDD ints)
    -> (n_frames, 10, H, W) calendar-month median composites.

    PASTIS spans Sep 2018 – Nov 2019, so observations from the same calendar
    month of both years are folded together; December has no acquisitions at
    all, so empty months fall back to the median of the 3 frames from the
    circularly nearest months.
    """
    months = np.array([int(str(int(d))[4:6]) for d in dates])  # 1-12
    frames = np.empty((n_frames, *s2.shape[1:]), dtype=np.float32)
    for m in range(1, n_frames + 1):
        sel = s2[months == m]
        if len(sel) == 0:
            dist = np.minimum(np.abs(months - m), 12 - np.abs(months - m))
            sel = s2[np.argsort(dist, kind="stable")[:3]]
        frames[m - 1] = np.median(sel, axis=0)
    return frames


@TRANSFORMS.register_module()
class LoadPASTISRasterTemporal(BaseTransform):
    """Load monthly-median S2 composites + 19-class semantic annotation.

    Outputs:
      * ``img``        – float32 (img_size, img_size, n_frames*10), normalised
      * ``gt_seg_map`` – int64 (img_size, img_size), 0-18 valid / 255 void
    """

    def __init__(self, img_size: int = 256, n_frames: int = 12):
        self.img_size = img_size
        self.n_frames = n_frames

    def transform(self, results: dict) -> dict:
        s2m_path = results.get("s2m_path")
        if s2m_path and Path(s2m_path).exists():
            frames = np.load(s2m_path).astype(np.float32)        # (12, 10, H, W)
        else:
            s2 = np.load(results["s2_path"]).astype(np.float32)  # (T, 10, H, W)
            frames = monthly_median(s2, results["dates_s2"], self.n_frames)
        assert frames.shape[0] == self.n_frames, \
            f"cache has {frames.shape[0]} frames, expected {self.n_frames}"

        frames = (frames - PASTIS_S2_MEAN[None, :, None, None]) \
            / PASTIS_S2_STD[None, :, None, None]

        T, C, ori_h, ori_w = frames.shape
        img = frames.reshape(T * C, ori_h, ori_w).transpose(1, 2, 0)  # (H, W, T*C)

        ann = np.load(results["ann_path"])[0].astype(np.uint8)   # (H, W), 0-19

        if self.img_size != ori_h or self.img_size != ori_w:
            # cv2.resize caps at 4 channels for some paths; resize per frame
            img = np.concatenate([
                cv2.resize(img[..., i * C:(i + 1) * C],
                           (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
                for i in range(T)
            ], axis=-1)
            ann = cv2.resize(ann, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)

        # U-TAE protocol: background (0) is a regular class, only void ignored
        gt_seg_map = ann.astype(np.int64)
        gt_seg_map[gt_seg_map >= 19] = 255

        H = W = self.img_size
        results["img"] = np.ascontiguousarray(img, dtype=np.float32)
        results["gt_seg_map"] = gt_seg_map
        results["img_shape"] = (H, W)
        results["ori_shape"] = (H, W)
        results["seg_fields"] = results.get("seg_fields", []) + ["gt_seg_map"]
        return results


@TRANSFORMS.register_module()
class PASTISRandomRotate90(BaseTransform):
    """Rotate img + seg maps by a random multiple of 90° (square inputs)."""

    def __init__(self, prob: float = 0.75):
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.rand() >= self.prob:
            return results
        k = np.random.randint(1, 4)
        results["img"] = np.ascontiguousarray(np.rot90(results["img"], k))
        for key in results.get("seg_fields", []):
            results[key] = np.ascontiguousarray(np.rot90(results[key], k))
        return results


@DATASETS.register_module()
class PASTISRasterTemporalDataset(BaseSegDataset):
    """PASTIS-R raster patches, 19-class (background + 18 crops) protocol."""

    METAINFO = dict(
        classes=[
            "background",
            "meadow", "soft_wheat", "corn", "sunflower", "sorghum",
            "barley", "dead_plant", "beet", "winter_peas", "winter_spelt",
            "grain_maze", "rapeseed", "beans", "peas", "hard_wheat",
            "triticale", "maize", "potato",
        ],
        palette=[[0, 0, 0]] + [[(i * 37) % 256, (i * 91) % 256, (i * 151) % 256]
                               for i in range(1, 19)],
    )

    def __init__(self, data_root: str, split: str = "train",
                 pipeline=None, **kwargs):
        self._pastis_root = Path(data_root)
        self._split = split
        if pipeline is None:
            pipeline = [
                dict(type="LoadPASTISRasterTemporal"),
                dict(type="PackSegInputs"),
            ]
        super().__init__(
            ann_file="", data_root="",
            img_suffix=".npy", seg_map_suffix=".npy",
            pipeline=pipeline,
            serialize_data=False, lazy_init=True,
            **kwargs,
        )
        self.data_list = self._build_data_list()
        self._fully_initialized = True

    def _build_data_list(self):
        with open(self._pastis_root / "metadata.geojson") as f:
            geojson = json.load(f)

        folds = _SPLIT_FOLDS[self._split]
        s2_dir = self._pastis_root / "DATA_S2"
        ann_dir = self._pastis_root / "ANNOTATIONS"
        cache_dir = self._pastis_root / CACHE_DIRNAME

        samples = []
        for feat in geojson["features"]:
            props = feat["properties"]
            if props["Fold"] not in folds:
                continue
            pid = int(props["ID_PATCH"])
            s2_path = s2_dir / f"S2_{pid}.npy"
            ann_path = ann_dir / f"TARGET_{pid}.npy"
            if not (s2_path.exists() and ann_path.exists()):
                continue
            dates = props["dates-S2"]
            if isinstance(dates, dict):
                dates = [dates[k] for k in sorted(dates, key=int)]
            samples.append({
                "s2_path": str(s2_path),
                "ann_path": str(ann_path),
                "s2m_path": str(cache_dir / f"S2M_{pid}.npy"),
                "dates_s2": dates,
            })
        return samples

    def load_data_list(self):
        return self._build_data_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Shallow-copy so pipeline outputs (31.5MB img per sample) are not
        # stored back into data_list — workers would otherwise cache the
        # whole dataset in RAM and get OOM-killed.
        return self.pipeline(dict(self.data_list[idx]))
