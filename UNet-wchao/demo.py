# import torch
# print(torch.cuda.is_available())  # 如果返回True，说明CUDA可用
# print(torch.__version__)  # 查看PyTorch版本

# import nibabel as nib
# print(nib.__version__)

# import torch
# print(f"PyTorch 版本: {torch.__version__}")
# print(f"CUDA 是否可用: {torch.cuda.is_available()}")
# print(f"GPU 数量: {torch.cuda.device_count()}")
# print(f"当前 GPU: {torch.cuda.get_device_name(0)}")

import nibabel as nib
import matplotlib
import numpy


print(nib.__version__)
print(numpy.__version__)
img = nib.load('')
data = img.get_fdata()