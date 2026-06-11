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
import cv2
import h5py

from mmcv.transforms import BaseTransform # type: ignore
from mmseg.registry import DATASETS, TRANSFORMS # type: ignore
from mmseg.datasets import BaseSegDataset # type: ignore

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

    def transform(self, results: dict) -> dict:
        import cv2

        # ================== 修改部分开始 ==================
        # 直接读取预处理后的 (H, W, 3) 轻量级数据
        img = np.load(results["s2_path"]).astype(np.float32) 
        ori_h, ori_w = img.shape[0], img.shape[1]
        # ================== 修改部分结束 ==================

        # TARGET has shape (3, H, W): channel 0 = semantic labels (0-19)
        ann = np.load(results["ann_path"])[0].astype(np.uint8)  # (H, W), values 0-19

        # Resize image and annotation if needed
        if self.img_size != ori_h or self.img_size != ori_w:
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
            ann = cv2.resize(ann, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)

        # Remap: shift crops 1-18 → 0-17; background (0) and void (≥19) → 255
        # gt_seg_map = ann.astype(np.int64) - 1        
        # gt_seg_map[gt_seg_map < 0] = 255             
        # gt_seg_map[gt_seg_map > 17] = 255            

        gt_seg_map = ann.astype(np.int64)                
        gt_seg_map[gt_seg_map > 18] = 255  

        H = W = self.img_size
        results["img"]        = img
        results["gt_seg_map"] = gt_seg_map
        results["img_shape"]  = (H, W)
        results["ori_shape"]  = (H, W)   
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

    # METAINFO = dict(
    #     classes=[f"crop_{i}" for i in range(1, 19)],   # 18 classes, 0-indexed
    #     palette=[[i * 13 % 256, i * 7 % 256, i * 17 % 256] for i in range(18)],
    # )

    METAINFO = dict(
        classes=["background"] + [f"crop_{i}" for i in range(1, 19)],
        palette=[[0, 0, 0]] + [[i * 13 % 256, i * 7 % 256, i * 17 % 256] for i in range(18)],
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
        # ================== 修改部分开始 ==================
        # 增加缓存机制：把有效文件路径存为 json 索引，避免在 NAS 上反复遍历
        cache_file = self._pastis_root / f"cache_{self._split}_mean_list.json"
        if cache_file.exists():
            # 第二次运行直接秒开读取
            with open(cache_file, 'r') as f:
                return json.load(f)

        # 第一次运行：解析 geojson
        with open(self._pastis_root / "metadata.geojson") as f:
            geojson = json.load(f)

        folds = _SPLIT_FOLDS[self._split]
        
        s2_dir  = self._pastis_root / "DATA_S2_RGB_MEAN"
        ann_dir = self._pastis_root / "ANNOTATIONS"

        samples = []
        for feat in geojson["features"]:
            props = feat["properties"]
            if props["Fold"] not in folds:
                continue
            pid = int(props["ID_PATCH"])
            s2_path  = s2_dir  / f"S2_{pid}.npy"
            ann_path = ann_dir / f"TARGET_{pid}.npy"
            
            # NAS上的 exists() 很慢，所以我们把成功的结果缓存下来
            if s2_path.exists() and ann_path.exists():
                samples.append({
                    "s2_path":  str(s2_path),
                    "ann_path": str(ann_path),
                })
                
        # 写入缓存文件
        with open(cache_file, 'w') as f:
            json.dump(samples, f)
            
        return samples

    def load_data_list(self):
        return self._build_data_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.pipeline(self.data_list[idx])