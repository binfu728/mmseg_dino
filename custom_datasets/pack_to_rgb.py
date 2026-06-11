"""预处理 PASTIS-R 数据: 从原始 S2 时序提取 12 帧 RGB, 拼接为 36 通道伪图像.

输出格式: (H, W, 36) float16 .npy 文件, 存于 DATA_S2_RGB_TIMESERIES/.

用法:
    python pack_to_rgb.py
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 请修改为你的 PASTIS-R 数据集所在的实际根目录
# ==========================================
pastis_root = Path('/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R')

s2_dir = pastis_root / "DATA_S2"
out_dir = pastis_root / "DATA_S2_RGB_TIMESERIES"
out_dir.mkdir(parents=True, exist_ok=True)

_RGB_IDX = [2, 1, 0]  # B04(Red), B03(Green), B02(Blue)
T_TARGET = 12          # OlmoEarth 设定的最大时间步数

s2_files = list(s2_dir.glob("S2_*.npy"))
print(f"Found {len(s2_files)} files. Start processing...")

for s2_path in tqdm(s2_files):
    out_path = out_dir / s2_path.name
    if out_path.exists():
        continue  # 跳过已处理的文件

    try:
        s2 = np.load(s2_path).astype(np.float32)  # (T, 10, H, W)
        s2_rgb = s2[:, _RGB_IDX, :, :]            # (T, 3, H, W)

        T_current = s2_rgb.shape[0]

        # 在时间维度均匀采样 12 帧
        if T_current >= T_TARGET:
            indices = np.linspace(0, T_current - 1, T_TARGET, dtype=int)
        else:
            # T_current < 12: 采样时允许重复 (np.linspace 会自动处理)
            indices = np.linspace(0, T_current - 1, T_TARGET, dtype=int)

        s2_sampled = s2_rgb[indices]  # (12, 3, H, W)

        # 变形为 (H, W, 36) 的伪图像, 兼容 mmseg 的 HWC 格式
        H, W = s2_sampled.shape[-2], s2_sampled.shape[-1]
        img_36 = s2_sampled.transpose(2, 3, 0, 1).reshape(H, W, T_TARGET * 3)

        # float16 节省一半磁盘空间, S2 反射率 0~10000 在 fp16 可表示范围内
        np.save(out_path, img_36.astype(np.float16))

    except Exception as e:
        print(f"Error processing {s2_path.name}: {e}")

print("Preprocessing Finished!")
