r"""Print train_results from best.pt.
"""

import torch
from pathlib import Path

ckpt_path = Path(r"D:\EcoSort-Resources\experiments\yolo11n_baseline_13class\weights\best.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print("train_results:", ckpt.get("train_results"))
print("train_metrics:", ckpt.get("train_metrics"))
