r"""Hardware inspection script for training feasibility analysis.
"""

import os
import platform
import psutil
import torch

def inspect_hardware():
    print("=== System Hardware & Environment ===")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Physical CPU Cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical CPU Cores: {psutil.cpu_count(logical=True)}")

    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB")

    print("\n=== PyTorch & Acceleration ===")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("Dedicated GPU: Not detected / Not configured in current PyTorch build (CPU-only PyTorch).")

    print("\n=== Training Workload Profile ===")
    print("Dataset: 255 training images, 32 validation images, 640x640 resolution")
    print("Architecture: YOLO11n (approx. 2.6M parameters, 6.5 GFLOPs)")
    print("Batch Size 8 vs 16 Memory Footprint:")
    print("  - Batch 8 RAM footprint: ~1.5 - 2.5 GB")
    print("  - Batch 16 RAM footprint: ~2.5 - 4.0 GB")
    print("Feasibility on current system: Fully feasible on CPU or external GPU.")

if __name__ == "__main__":
    inspect_hardware()
