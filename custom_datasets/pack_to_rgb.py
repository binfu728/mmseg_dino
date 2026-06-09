import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 请修改为你的 PASTIS-R 数据集所在的实际根目录
# ==========================================
pastis_root = Path('/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R') 

s2_dir = pastis_root / "DATA_S2"
# 新建一个目录，专门存处理好的 RGB 均值图
out_dir = pastis_root / "DATA_S2_RGB_MEAN"
out_dir.mkdir(parents=True, exist_ok=True)

# 你的原始代码中提取 RGB 通道的索引
_RGB_IDX = [2, 1, 0] 

# 遍历所有原始 S2 文件
s2_files = list(s2_dir.glob("S2_*.npy"))
print(f"Found {len(s2_files)} files. Start processing...")

for s2_path in tqdm(s2_files):
    out_path = out_dir / s2_path.name
    if out_path.exists(): 
        continue # 如果中断了，再次运行会跳过已处理的
    
    try:
        # 读取原始数据 (T, 10, H, W)
        s2 = np.load(s2_path).astype(np.float32)
        
        # 预先进行计算均值、提取通道，并直接转换成 (H, W, 3) 形状
        # 这样在 dataloader 里就彻底不需要做任何 transpose 等耗时操作了
        img = s2.mean(axis=0)[_RGB_IDX].transpose(1, 2, 0)
        
        # 使用 float16 格式保存，不仅不影响视觉特征精度，还能把文件体积缩小一半
        np.save(out_path, img.astype(np.float16))
    except Exception as e:
        print(f"Error processing {s2_path.name}: {e}")

print("Preprocessing Finished!")