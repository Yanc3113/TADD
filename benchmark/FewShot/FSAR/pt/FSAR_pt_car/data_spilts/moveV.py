# import os
# import shutil

# # 定义视频所在的目录和帧目录的根路径
# video_root = r"C:\Users\12787\Desktop\Car\RGB_20s"
# frame_root = r"C:\Futures\Knowledge-Prompting-for-FSAR\data_car\RGB_20s_frame_fps_5"

# def move_videos_to_frame_folders(video_root, frame_root):
#     # 遍历视频目录
#     for root, dirs, files in os.walk(video_root):
#         for file in files:
#             if file.endswith('.mp4'):  # 如果是.mp4文件
#                 video_path = os.path.join(root, file)  # 获取视频完整路径

#                 # 获取视频相对路径
#                 relative_path = os.path.relpath(video_path, video_root)
                
#                 # 去除视频文件名的扩展名以匹配帧目录
#                 video_folder = os.path.splitext(relative_path)[0]

#                 # 构造对应的帧目录路径
#                 frame_folder = os.path.join(frame_root, video_folder)

#                 # 检查帧目录是否存在
#                 if os.path.exists(frame_folder):
#                     print(f"Moving video {video_path} to {frame_folder}...")
                    
#                     # 将视频剪切到对应的帧目录
#                     try:
#                         shutil.move(video_path, os.path.join(frame_folder, file))
#                         print(f"Successfully moved {video_path} to {frame_folder}")
#                     except Exception as e:
#                         print(f"Failed to move {video_path}: {e}")
#                 else:
#                     print(f"Frame directory {frame_folder} does not exist. Skipping {video_path}.")

# # 调用函数，开始移动视频文件
# move_videos_to_frame_folders(video_root, frame_root)











import os
import shutil

# 定义帧目录的根路径
frame_root = r"C:\Futures\Knowledge-Prompting-for-FSAR\data_car\RGB_20s_frame_fps_5"

def move_videos_to_parent_folder(frame_root):
    # 遍历帧目录
    for root, dirs, files in os.walk(frame_root):
        for file in files:
            if file.endswith('.mp4'):  # 如果是.mp4文件
                video_path = os.path.join(root, file)  # 获取视频完整路径

                # 构造视频应该移动到的父目录路径
                parent_folder = os.path.dirname(root)  # 获取当前帧目录的父目录
                video_name = os.path.basename(root) + '.mp4'  # 构造新的视频文件名（例如 video_24_1.mp4）
                new_video_path = os.path.join(parent_folder, video_name)  # 构造新的视频路径

                # 移动视频文件到父目录
                print(f"Moving video {video_path} to {new_video_path}...")
                try:
                    shutil.move(video_path, new_video_path)
                    print(f"Successfully moved {video_path} to {new_video_path}")
                except Exception as e:
                    print(f"Failed to move {video_path}: {e}")

# 调用函数，开始移动视频文件
move_videos_to_parent_folder(frame_root)

