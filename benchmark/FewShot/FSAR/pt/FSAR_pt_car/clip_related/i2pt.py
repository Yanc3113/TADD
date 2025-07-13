import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

model.eval()

def extract_features_from_frames(frame_dir, batch_size=100):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    features_list = []

    for i in range(0, len(frame_files), batch_size):
        frames = []
        for frame_file in frame_files[i:i + batch_size]:
            frame_path = os.path.join(frame_dir, frame_file)
            image = Image.open(frame_path).convert("RGB")
            frames.append(image)
        
        # Process the current batch
        inputs = processor(images=frames, return_tensors="pt", padding=True)
        with torch.no_grad():
            batch_features = model.get_image_features(**inputs)
        
        features_list.append(batch_features)

    # Concatenate all features into a single tensor
    features = torch.cat(features_list, dim=0)
    return features

def process_all_videos(video_root_dir, output_dir):
    for root, dirs, files in os.walk(video_root_dir):
        if any(file.endswith(".png") for file in files):
            relative_path = os.path.relpath(root, video_root_dir)
            category, video_folder = os.path.split(relative_path)
            features = extract_features_from_frames(root, batch_size=150)

            save_dir = os.path.join(output_dir, category)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            output_file = os.path.join(save_dir, f"{video_folder}.pt")
            torch.save(features, output_file)
            print(f"Saved features for {video_folder} in category {category} to {output_file}")

video_root_dir = r"D:\Data\Frame_Plat_fps_5" # 包含所有视频帧的根目录
output_dir = r"C:\Futures\Knowledge-Prompting-for-FSAR\data_plat\clip_related\VitOutput_plat" # 保存提取特征的目录

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

process_all_videos(video_root_dir, output_dir)
