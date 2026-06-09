"""PASTIS / PASTIS-R dataset adapters for mmsegmentation.

Two formats are supported:

PixelSet format (路径B):
    Each parcel is represented as a pseudo-2D image: its N pixels are randomly
    sampled (with replacement when N < grid_size²) and arranged into a square
    grid.  All grid positions share the same crop-type label — a
    classification-proxy task usable with standard segmentation heads.

    Data layout:
        <data_root>/
            metadata_parcel.csv      # columns: ID_PARCEL, ID_PATCH, Label, Fold
            DATA_S2/
                S2_<ID_PARCEL>.npy   # float/int16 array  [T, 10, N]
            NORM_PARCEL_S2_set.json  # per-fold channel mean/std

Raster Patch format (路径C):
    Each spatial patch (128×128 pixels) is loaded as a real 2D image with a
    per-pixel semantic annotation.  Background pixels (original label 0) and
    any out-of-range labels (e.g. 19) are mapped to ignore_index=255;
    crop labels 1-18 are shifted to 0-17.

    Data layout (PASTIS-R, zenodo.org/records/5735646):
        <data_root>/
            metadata.geojson         # GeoJSON FeatureCollection; each feature has
                                     #   properties: ID_PATCH (int), Fold (1-5), ...
            DATA_S2/
                S2_<PATCH_ID>.npy    # int16 array  [T, 10, H, W]
            ANNOTATIONS/
                TARGET_<PATCH_ID>.npy  # uint8 array  [3, H, W]
                                       #   channel 0: semantic labels 0-19
                                       #   channel 1: instance IDs
                                       #   channel 2: panoptic labels

Folds: 1-3 → train, 4 → val, 5 → test  (canonical PASTIS split)
Classes: 18 crop types (0-indexed 0-17); background (0) and void (19) → ignore_index=255.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from mmcv.transforms import BaseTransform
from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.datasets import BaseSegDataset

# RGB band indices within the 10-band S2 stack
# Band order in .npy:  B02, B03, B04, B05, B06, B07, B08, B08A, B11, B12
# RGB = B04 (Red, idx 2), B03 (Green, idx 1), B02 (Blue, idx 0)
_RGB_IDX = [2, 1, 0]

# PixelSet normalisation stats for RGB bands (R, G, B) — from NORM_PARCEL_S2_set.json
# Used by the mmseg SegDataPreProcessor; values are raw S2 reflectance (~0–10000).
PASTIS_RGB_MEAN = [994.4, 931.4, 876.4]
PASTIS_RGB_STD  = [2053.8, 1976.2, 2049.3]

# Raster Patch normalisation stats for RGB bands (R=B04, G=B03, B=B02)
# Computed from NORM_S2_patch.json by averaging Fold_1..5 then extracting RGB indices.
PASTIS_RASTER_RGB_MEAN = [1436.7, 1387.7, 1180.2]
PASTIS_RASTER_RGB_STD  = [1996.2, 1916.8, 1976.7]

_SPLIT_FOLDS = {"train": [1, 2, 3], "val": [4], "test": [5]}


@TRANSFORMS.register_module()
class LoadPASTISPixelSet(BaseTransform):
    """Load a PASTIS PixelSet parcel and produce a pseudo-2D image.

    Reads ``results['s2_path']`` (path to S2_*.npy) and ``results['label']``
    (int, 1-18).  Outputs the standard mmseg keys:

    * ``img``        – float32 np.ndarray [H, W, 3], raw reflectance (R, G, B),
                       NOT normalised (SegDataPreProcessor handles that).
    * ``gt_seg_map`` – int64 np.ndarray [H, W], constant label 0-17.
    * ``img_shape``, ``ori_shape`` – (H, W).

    The parcel's N pixels are randomly sampled with replacement to fill
    H × W positions, then shuffled, so each call produces a different layout.

    Args:
        grid_size (int): output H = W.  Default 64 → 64×64 pseudo-image.
    """

    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.n_pixel = grid_size * grid_size

    def transform(self, results: dict) -> dict:
        s2 = np.load(results["s2_path"]).astype(np.float32)  # (T, C=10, N)

        # Temporal mean
        img = s2.mean(axis=0)  # (C=10, N)
        N = img.shape[-1]
        n = self.n_pixel

        # Sample n pixels; repeat smaller parcels to fill the grid
        if N >= n:
            idx = np.random.choice(N, n, replace=False)
        else:
            idx = np.concatenate([
                np.arange(N),
                np.random.choice(N, n - N, replace=True),
            ])
            np.random.shuffle(idx)

        img = img[_RGB_IDX][:, idx]  # (3, n)  →  R, G, B

        H = W = self.grid_size
        img = img.reshape(3, H, W).transpose(1, 2, 0).copy()  # (H, W, 3)

        # Parcel label: shift 1-18 → 0-17
        label = int(results["label"]) - 1
        gt_seg_map = np.full((H, W), label, dtype=np.int64)

        results["img"] = img
        results["gt_seg_map"] = gt_seg_map
        results["img_shape"] = (H, W)
        results["ori_shape"] = (H, W)
        results["seg_fields"] = results.get("seg_fields", []) + ["gt_seg_map"]
        return results


@DATASETS.register_module()
class PASTISDataset(BaseSegDataset):
    """PASTIS PixelSet crop-type dataset wrapped for mmsegmentation.

    Each parcel in the metadata is one sample.  The ``LoadPASTISPixelSet``
    transform turns it into a pseudo-image for use with any mmseg head.

    Args:
        data_root (str): root directory of PASTIS-R_PixelSet
                         (must contain ``metadata_parcel.csv`` and ``DATA_S2/``).
        split     (str): ``'train'`` | ``'val'`` | ``'test'``.
        pipeline  (list): mmseg transform pipeline.
    """

    METAINFO = dict(
        classes=[f"crop_{i}" for i in range(1, 19)],   # 18 classes, 0-indexed
        palette=[[i * 13 % 256, i * 7 % 256, i * 17 % 256] for i in range(18)],
    )

    def __init__(self, data_root: str, split: str = "train",
                 pipeline=None, **kwargs):
        self._pastis_root = Path(data_root)
        self._split = split
        if pipeline is None:
            pipeline = [
                dict(type="LoadPASTISPixelSet"),
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
        meta = pd.read_csv(self._pastis_root / "metadata_parcel.csv")
        folds = _SPLIT_FOLDS[self._split]
        meta = meta[meta["Fold"].isin(folds)].copy()
        s2_dir = self._pastis_root / "DATA_S2"
        samples = []
        for _, row in meta.iterrows():
            pid = int(row["ID_PARCEL"])
            s2_path = s2_dir / f"S2_{pid}.npy"
            if s2_path.exists():
                samples.append({
                    "s2_path": str(s2_path),
                    "label": int(row["Label"]),
                })
        return samples

    # BaseSegDataset interface
    def load_data_list(self):
        return self._build_data_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.pipeline(self.data_list[idx])


# ---------------------------------------------------------------------------
# Raster Patch format (路径C) — true pixel-level semantic segmentation
# ---------------------------------------------------------------------------

@TRANSFORMS.register_module()
class LoadPASTISRaster(BaseTransform):
    """Load a PASTIS-R raster patch and its per-pixel semantic annotation.

    Reads ``results['s2_path']`` (S2_*.npy, shape [T, 10, H, W]) and
    ``results['ann_path']`` (TARGET_*.npy, shape [H, W], values 0-18).

    Processing steps:
      1. Temporal mean over T → (10, H, W)
      2. Select RGB bands (B04, B03, B02 → indices 2, 1, 0) → (3, H, W)
      3. Transpose to HWC → (H, W, 3), raw reflectance, NOT normalised
      4. Resize image to ``img_size × img_size`` with bilinear interpolation
      5. Resize annotation with nearest-neighbour interpolation
      6. Remap labels: 1-18 → 0-17 (crops);  0 → 255 (background / ignore)

    Outputs the standard mmseg keys:
      * ``img``        – float32 np.ndarray [img_size, img_size, 3]
      * ``gt_seg_map`` – int64   np.ndarray [img_size, img_size], 0-17 or 255
      * ``img_shape``, ``ori_shape``

    Args:
        img_size (int): output H = W after resize.  Default 512.
    """

    def __init__(self, img_size: int = 512):
        self.img_size = img_size

    # def transform(self, results: dict) -> dict:
    #     import cv2

    #     s2 = np.load(results["s2_path"]).astype(np.float32)   # (T, 10, H, W)
    #     ori_h, ori_w = s2.shape[-2], s2.shape[-1]

    #     # Temporal mean → select RGB bands → HWC
    #     img = s2.mean(axis=0)[_RGB_IDX].transpose(1, 2, 0).copy()  # (H, W, 3)

    #     # TARGET has shape (3, H, W): channel 0 = semantic labels (0-19)
    #     ann = np.load(results["ann_path"])[0].astype(np.uint8)  # (H, W), values 0-19

    #     # Resize image and annotation if needed
    #     if self.img_size != ori_h or self.img_size != ori_w:
    #         img = cv2.resize(img, (self.img_size, self.img_size),
    #                          interpolation=cv2.INTER_LINEAR)
    #         ann = cv2.resize(ann, (self.img_size, self.img_size),
    #                          interpolation=cv2.INTER_NEAREST)

    #     # Remap: shift crops 1-18 → 0-17; background (0) and void (≥19) → 255
    #     gt_seg_map = ann.astype(np.int64) - 1        # 0-17 for crops, -1 for bg, 18 for label=19
    #     gt_seg_map[gt_seg_map < 0] = 255             # background → ignore
    #     gt_seg_map[gt_seg_map > 17] = 255            # void/out-of-range → ignore

    #     H = W = self.img_size
    #     results["img"]        = img
    #     results["gt_seg_map"] = gt_seg_map
    #     results["img_shape"]  = (H, W)
    #     results["ori_shape"]  = (H, W)   # both GT and pred live at img_size; avoid postprocess downscale
    #     results["seg_fields"] = results.get("seg_fields", []) + ["gt_seg_map"]
    #     return results

    def transform(self, results: dict) -> dict:
        import cv2
        import h5py

        # 【重点修复多进程卡死】在 worker 内部懒加载打开 HDF5，防止多进程冲突
        if not hasattr(self, 'h5f'):
            self.h5f = h5py.File(results["h5_path"], 'r',swmr=True)
            
        pid = results["pid"]
        
        # 直接从 HDF5 中以极速切片读取数据，替代 np.load()
        s2 = self.h5f['DATA_S2'][pid][:]
        ann_raw = self.h5f['ANNOTATIONS'][pid][:]
        
        # ----------- 以下原封不动 -----------
        s2 = s2.astype(np.float32)
        ori_h, ori_w = s2.shape[-2], s2.shape[-1]
        img = s2.mean(axis=0)[_RGB_IDX].transpose(1, 2, 0).copy()
        ann = ann_raw[0].astype(np.uint8)

        if self.img_size != ori_h or self.img_size != ori_w:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            ann = cv2.resize(ann, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        gt_seg_map = ann.astype(np.int64) - 1
        gt_seg_map[gt_seg_map < 0] = 255
        gt_seg_map[gt_seg_map > 17] = 255

        results["img"] = img
        results["gt_seg_map"] = gt_seg_map
        results["img_shape"] = (self.img_size, self.img_size)
        results["ori_shape"] = (self.img_size, self.img_size)
        results["seg_fields"] = results.get("seg_fields", []) + ["gt_seg_map"]
        return results


@DATASETS.register_module()
class PASTISRasterDataset(BaseSegDataset):
    """PASTIS-R Raster Patch crop-type dataset for mmsegmentation.

    Loads 128×128 spatial patches from the official PASTIS-R raster release
    (zenodo.org/records/5735646).  Background (original label 0) is
    mapped to ignore_index=255; crop classes 1-18 are shifted to 0-17.

    Args:
        data_root (str): root directory of PASTIS-R
                         (must contain ``metadata.json``, ``DATA_S2/``,
                         and ``ANNOTATIONS/``).
        split     (str): ``'train'`` | ``'val'`` | ``'test'``.
        pipeline  (list): mmseg transform pipeline.
    """

    METAINFO = dict(
        classes=[f"crop_{i}" for i in range(1, 19)],   # 18 classes, 0-indexed
        palette=[[i * 13 % 256, i * 7 % 256, i * 17 % 256] for i in range(18)],
    )

    def __init__(self, data_root: str, split: str = "train",
                 pipeline=None, **kwargs):
        self._pastis_root = Path(data_root)
        self._split = split
        if pipeline is None:
            pipeline = [
                dict(type="LoadPASTISRaster"),
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

    # def _build_data_list(self):
    #     # PASTIS-R uses metadata.geojson (GeoJSON FeatureCollection).
    #     # Each feature's properties contain: ID_PATCH (int), Fold (1-5), ...
    #     with open(self._pastis_root / "metadata.geojson") as f:
    #         geojson = json.load(f)

    #     folds = _SPLIT_FOLDS[self._split]
    #     s2_dir  = self._pastis_root / "DATA_S2"
    #     ann_dir = self._pastis_root / "ANNOTATIONS"

    #     samples = []
    #     for feat in geojson["features"]:
    #         props = feat["properties"]
    #         if props["Fold"] not in folds:
    #             continue
    #         pid = int(props["ID_PATCH"])
    #         s2_path  = s2_dir  / f"S2_{pid}.npy"
    #         ann_path = ann_dir / f"TARGET_{pid}.npy"
    #         if s2_path.exists() and ann_path.exists():
    #             samples.append({
    #                 "s2_path":  str(s2_path),
    #                 "ann_path": str(ann_path),
    #             })
    #     return samples

    def _build_data_list(self):
        with open(self._pastis_root / "metadata.geojson") as f:
            geojson = json.load(f)
        folds = _SPLIT_FOLDS[self._split]
        
        # 【新增】告诉数据集 HDF5 文件在哪
        h5_path = str(self._pastis_root.parent / "pastis_raster.h5") 
        
        samples = []
        for feat in geojson["features"]:
            props = feat["properties"]
            if props["Fold"] not in folds:
                continue
            pid = str(props["ID_PATCH"])
            # 仅仅保存 ID 和 HDF5 的位置
            samples.append({
                "pid": pid,
                "h5_path": h5_path
            })
        return samples

    def load_data_list(self):
        return self._build_data_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.pipeline(self.data_list[idx])
