"""PASTIS-R temporal dataset: monthly median composite + 19-class U-TAE protocol.

12-month median compositing: S2 ~43 irregular dates → 12 fixed monthly frames.
Each frame = median of all observations in that calendar month → (12, 10, H, W).
Stacked to (H, W, 120) for mmseg pipeline compatibility.

Folds: 1-3 → train, 4 → val, 5 → test  (canonical PASTIS split)
"""
from pathlib import Path
import json

import numpy as np
import cv2

from mmcv.transforms import BaseTransform
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmengine.registry import TRANSFORMS
from mmseg.registry import DATASETS

_S2_10B_MEAN = [1180.2, 1387.7, 1436.7, 1773.7, 2735.8, 3080.1, 3223.6, 3338.3, 2418.1, 1630.2]
_S2_10B_STD  = [1976.7, 1916.8, 1996.2, 1903.1, 1784.9, 1796.3, 1811.8, 1793.3, 1474.4, 1309.8]

_SPLIT_FOLDS = {"train": [1, 2, 3], "val": [4], "test": [5]}


@TRANSFORMS.register_module()
class LoadPASTISRasterTemporal(BaseTransform):
    def __init__(self, img_size: int = 256, num_classes: int = 19, n_frames: int = 12):
        self.img_size = img_size
        self.num_classes = num_classes
        self.n_frames = n_frames

    def _compute_monthly_median(self, s2, dates_list):
        months = np.array([int(str(d)[4:6]) for d in dates_list], dtype=np.int32)
        frames = []
        for m in range(1, 13):
            sel = s2[months == m]
            if len(sel) == 0:
                indices = np.abs(months - m).argsort()[:3]
                sel = s2[indices]
            frames.append(np.median(sel, axis=0))
        return np.stack(frames)

    def transform(self, results: dict) -> dict:
        cache_path = results.get("s2m_path")
        if cache_path and Path(cache_path).exists():
            frames = np.load(cache_path).astype(np.float32) # (12, 10, H, W)
        else:
            s2 = np.load(results["s2_path"]).astype(np.float32)
            dates_list = results["dates_s2"]
            frames = self._compute_monthly_median(s2, dates_list)

        ori_h, ori_w = 128, 128
        
        # 归一化
        mean = np.array(_S2_10B_MEAN, dtype=np.float32).reshape(1, -1, 1, 1)
        std  = np.array(_S2_10B_STD,  dtype=np.float32).reshape(1, -1, 1, 1)
        frames = (frames - mean) / std

        # 转换为 (H, W, T*C) 准备被 resize
        T, C, H, W = frames.shape
        img = frames.reshape(T * C, ori_h, ori_w).transpose(1, 2, 0) # (128, 128, 120)

        ann = np.load(results["ann_path"])[0].astype(np.uint8)

        if self.img_size != ori_h or self.img_size != ori_w:
            # 修复：按帧拆分进行 resize，避免 opencv 通道限制导致底层崩溃
            img = np.concatenate([
                cv2.resize(img[..., i * C:(i + 1) * C],
                           (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
                for i in range(T)
            ], axis=-1)
            ann = cv2.resize(ann, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)

        gt_seg_map = ann.astype(np.int64)
        gt_seg_map[gt_seg_map >= 19] = 255

        results["img"]        = np.ascontiguousarray(img, dtype=np.float32)
        results["gt_seg_map"] = gt_seg_map
        results["img_shape"]  = (self.img_size, self.img_size)
        results["ori_shape"]  = (self.img_size, self.img_size)
        results["seg_fields"] = results.get("seg_fields", []) + ["gt_seg_map"]
        return results


# 修复：补齐缺失的旋转数据增强，防止启动报错
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
        s2_dir  = self._pastis_root / "DATA_S2"
        s2m_dir = self._pastis_root / "DATA_S2_M12"
        ann_dir = self._pastis_root / "ANNOTATIONS"
        samples = []
        for feat in geojson["features"]:
            props = feat["properties"]
            if props["Fold"] not in folds:
                continue
            pid = int(props["ID_PATCH"])
            s2_path  = s2_dir  / f"S2_{pid}.npy"
            s2m_path = s2m_dir / f"S2M_{pid}.npy"
            ann_path = ann_dir / f"TARGET_{pid}.npy"
            if not s2_path.exists() or not ann_path.exists():
                continue
            dates_raw = props["dates-S2"]
            dates_list = [int(v) for v in (dates_raw.values() if isinstance(dates_raw, dict) else dates_raw)]
            samples.append({
                "s2_path":   str(s2_path),
                "s2m_path":  str(s2m_path),
                "ann_path":  str(ann_path),
                "dates_s2":  dates_list,
            })
        return samples

    def load_data_list(self):
        return self._build_data_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 避免字典直接缓存导致的OOM
        return self.pipeline(dict(self.data_list[idx]))