import torch
print("CUDA available:", torch.cuda.is_available())  # 检查CUDA是否可用
print("CUDA version:", torch.version.cuda)  # 输出CUDA版本
print("GPU count:", torch.cuda.device_count())  # 检查可用GPU数量
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")  # 输出GPU名称
