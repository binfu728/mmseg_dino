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

    def transform(self, results: dict) -> dict:
        import cv2
        import h5py

        # 【重点修复多进程卡死】在 worker 内部懒加载打开 HDF5，防止多进程冲突
        if not hasattr(self, 'h5f'):
            self.h5f = h5py.File(results["h5_path"], 'r')
            
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