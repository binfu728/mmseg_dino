import os
import json
import numpy as np
import h5py
from tqdm import tqdm

# 你原来的 NAS 数据路径
PASTIS_ROOT = '/mnt/ht2-nas2/00-model/00-fb/mmseg_data/PASTIS-R'
OUTPUT_H5 = '/mnt/ht2-nas2/00-model/00-fb/mmseg_data/pastis_raster.h5'

print("⏳ 正在将碎裂的 .npy 打包进 HDF5 数据库...")
with open(os.path.join(PASTIS_ROOT, 'metadata.geojson')) as f:
    geojson = json.load(f)

# 创建单个大文件
with h5py.File(OUTPUT_H5, 'w') as h5f:
    grp_s2 = h5f.create_group('DATA_S2')
    grp_ann = h5f.create_group('ANNOTATIONS')
    
    for feat in tqdm(geojson['features']):
        pid = str(feat['properties']['ID_PATCH'])
        s2_path = os.path.join(PASTIS_ROOT, 'DATA_S2', f'S2_{pid}.npy')
        ann_path = os.path.join(PASTIS_ROOT, 'ANNOTATIONS', f'TARGET_{pid}.npy')
        
        if os.path.exists(s2_path) and os.path.exists(ann_path):
            # 将 numpy 数据直接存入 HDF5
            grp_s2.create_dataset(pid, data=np.load(s2_path), compression="lzf")
            grp_ann.create_dataset(pid, data=np.load(ann_path), compression="lzf")

print(f"✅ 打包完成！所有数据已凝聚为一个巨型文件: {OUTPUT_H5}")