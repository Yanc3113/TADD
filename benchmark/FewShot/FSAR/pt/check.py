import torch

def load_and_print_pt_file(file_path):
    """加载并打印单个 pt 文件的内容"""
    try:
        data = torch.load(file_path)
        print(f"File: {file_path}")
        print("Data type:", type(data))
        print("Data shape:" if hasattr(data, "shape") else "Data length:", data.shape if hasattr(data, "shape") else len(data))
        print("Data content (first 5 elements):", data[:5] if hasattr(data, "__getitem__") else data)
        print("=" * 50)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# 设置两个 pt 文件的路径
success_file_path = r"C:\Futures\Knowledge-Prompting-for-FSAR\data_car\clip_related\VitOutput_RGB_20s_frame_fps_5\Back_Seat_Passenger_Leaning_Out_of_Window\video_24_1.pt"
fail_file_path = r"C:\Futures\Knowledge-Prompting-for-FSAR\data_plat\clip_related\VitOutput_plat\chat_block_blindway\video_23.pt"

# 加载并打印两个文件内容
print("Printing contents of the first .pt file (success file):")
load_and_print_pt_file(success_file_path)

print("\nPrinting contents of the second .pt file (fail file):")
load_and_print_pt_file(fail_file_path)
