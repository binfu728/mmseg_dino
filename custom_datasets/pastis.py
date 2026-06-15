"""PASTIS-R Raster Patch dataset adapter for mmsegmentation.

Supports 10-band Sentinel-2 input with per-band normalization and
19-class U-TAE protocol (background = class 0, void = 255).

Folds: 1-3 → train, 4 → val, 5 → test  (canonical PASTIS split)
"""
from pathlib import Path
import json

import numpy as np

from mmcv.transforms import BaseTransform
<<<<<<< HEAD
from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.registry import DATASETS

# RGB band indices within the 10-band S2 stack
# Band order in .npy:  B02, B03, B04, B05, B06, B07, B08, B08A, B11, B12
# RGB = B04 (Red, idx 2), B03 (Green, idx 1), B02 (Blue, idx 0)
_RGB_IDX = [2, 1, 0]
=======
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmengine.registry import TRANSFORMS
from mmseg.registry import DATASETS
>>>>>>> 956313c76c03b586186bab5e4b98bfc2eb4d2585

_S2_10B_MEAN = [1180.2, 1387.7, 1436.7, 1773.7, 2735.8, 3080.1, 3223.6, 3338.3, 2418.1, 1630.2]
_S2_10B_STD  = [1976.7, 1916.8, 1996.2, 1903.1, 1784.9, 1796.3, 1811.8, 1793.3, 1474.4, 1309.8]

_SPLIT_FOLDS = {"train": [1, 2, 3], "val": [4], "test": [5]}


@TRANSFORMS.register_module()
class LoadPASTISRaster(BaseTransform):
    def __init__(self, img_size: int = 256, num_classes: int = 19):
        self.img_size = img_size
        self.num_classes = num_classes

    def transform(self, results: dict) -> dict:
        import cv2

        s2 = np.load(results["s2_path"]).astype(np.float32)
        ori_h, ori_w = s2.shape[-2], s2.shape[-1]

        img = s2.mean(axis=0).transpose(1, 2, 0).copy()

        mean = np.array(_S2_10B_MEAN, dtype=np.float32)
        std  = np.array(_S2_10B_STD, dtype=np.float32)
        img = (img - mean) / std

        ann = np.load(results["ann_path"])[0].astype(np.uint8)

        if self.img_size != ori_h or self.img_size != ori_w:
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
            ann = cv2.resize(ann, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)

        gt_seg_map = ann.astype(np.int64)
        gt_seg_map[gt_seg_map == 19] = 255

        H = W = self.img_size
        results["img"]        = img
        results["gt_seg_map"] = gt_seg_map
        results["img_shape"]  = (H, W)
        results["ori_shape"]  = (H, W)
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
class PASTISRasterDataset(BaseSegDataset):

    METAINFO = dict(
        classes=[f"crop_{i}" for i in range(1, 20)],
        palette=[[i * 13 % 256, i * 7 % 256, i * 17 % 256] for i in range(19)],
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

    def _build_data_list(self):
        with open(self._pastis_root / "metadata.geojson") as f:
            geojson = json.load(f)
        folds = _SPLIT_FOLDS[self._split]
        s2_dir  = self._pastis_root / "DATA_S2"
        ann_dir = self._pastis_root / "ANNOTATIONS"
        samples = []
        for feat in geojson["features"]:
            props = feat["properties"]
            if props["Fold"] not in folds:
                continue
            pid = int(props["ID_PATCH"])
            s2_path  = s2_dir  / f"S2_{pid}.npy"
            ann_path = ann_dir / f"TARGET_{pid}.npy"
            if s2_path.exists() and ann_path.exists():
                samples.append({
                    "s2_path":  str(s2_path),
                    "ann_path": str(ann_path),
                })
        return samples

    def load_data_list(self):
        return self._build_data_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.pipeline(dict(self.data_list[idx]))
