import torch
import mmcv
import mmdet
import mmseg
from mmcv.ops import MultiScaleDeformableAttention

print("="*50)
print(f"🔥 PyTorch 版本: {torch.__version__}, CUDA 是否可用: {torch.cuda.is_available()}")
print(f"📦 MMCV 版本: {mmcv.__version__}")
print(f"📦 MMDetection 版本: {mmdet.__version__}")
print(f"📦 MMSegmentation 版本: {mmseg.__version__}")
print("="*50)

# 最核心的验证：测试 MMCV 底层 C++ 算子是否能跑通
print("⏳ 正在测试 MMCV CUDA 算子 (MSDA)...")
try:
    # 尝试在 GPU 上初始化 MSDA 算子
    msda = MultiScaleDeformableAttention(embed_dims=256, num_heads=8, num_levels=4, num_points=4).cuda()
    print("✅ 太牛了！MMCV CUDA 算子编译完美，加载成功！环境毫无问题！")
except Exception as e:
    print("❌ MMCV CUDA 算子加载失败，可能编译有问题，错误信息：\n", e)