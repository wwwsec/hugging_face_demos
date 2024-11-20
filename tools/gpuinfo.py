# 检查GPU信息
import torch

def check_gpu_support():
    if torch.cuda.is_available():
        print("GPU is available.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("GPU is not available.")

check_gpu_support()
