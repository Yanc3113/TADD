import torch
import os
import glob

# 指定 .pt 文件所在的根文件夹路径
proposals_fea_folder = r"C:\Futures\Knowledge-Prompting-for-FSAR\data_plat\clip_related\VitOutput_plat"
output_file = r"C:\Futures\Knowledge-Prompting-for-FSAR\data_plat\clip_related\combined_proposals_fea.pt"

# 获取所有子目录中的 .pt 文件路径
all_fea_files = glob.glob(os.path.join(proposals_fea_folder, "**", "*.pt"), recursive=True)

# 用于存储所有特征
combined_features = []

# 读取每个 .pt 文件并将其特征加载进来
for fea_file in all_fea_files:
    fea = torch.load(fea_file)
    combined_features.append(fea)

# 将所有特征堆叠在一起
# combined_features = torch.stack(combined_features)
combined_features = torch.cat(combined_features, dim=0)


# 保存为单个 .pt 文件
torch.save(combined_features, output_file)

print(f"Saved combined proposals features to {output_file}")
