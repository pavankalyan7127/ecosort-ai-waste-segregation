r"""EcoSort AI — YOLO11n-P2 1-Epoch Training Feasibility Preflight Script.
Executes exactly one epoch at imgsz=640, batch=2 on CPU.
STRICT CONSTRAINT: Only 1 epoch. No full training.
"""

import time
import os
import psutil
import json
from pathlib import Path
from ultralytics import YOLO

YAML_PATH = Path(r"ecosort-ai\models\yolo11n-p2.yaml")
PRETRAINED_WEIGHTS = Path(r"D:\EcoSort-Resources\models\pretrained\yolo11n.pt")
DATA_YAML = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\data.yaml")
EXPERIMENTS_DIR = Path(r"D:\EcoSort-Resources\experiments")
EXP_NAME = "yolo11n_p2_preflight"
EXP_DIR = EXPERIMENTS_DIR / EXP_NAME

def main():
    print("=== Step 1: Pre-Training Feasibility Verification ===")
    assert YAML_PATH.exists(), f"Missing YAML at {YAML_PATH}"
    assert PRETRAINED_WEIGHTS.exists(), f"Missing pretrained weights at {PRETRAINED_WEIGHTS}"
    assert DATA_YAML.exists(), f"Missing data.yaml at {DATA_YAML}"
    
    mem_before = psutil.virtual_memory()
    disk_before = psutil.disk_usage("D:\\")
    print(f"Host Available RAM: {mem_before.available / (1024**3):.2f} GB / {mem_before.total / (1024**3):.2f} GB")
    print(f"D: Drive Space: {disk_before.free / (1024**3):.2f} GB free")
    
    process = psutil.Process()
    initial_rss = process.memory_info().rss / (1024**2)
    
    print("\n=== Step 2: Instantiating YOLO11n-P2 & Loading Pretrained Weights ===")
    model = YOLO(str(YAML_PATH))
    model.load(str(PRETRAINED_WEIGHTS))
    
    print("\n=== Step 3: Executing 1-Epoch Feasibility Preflight ===")
    print(f"Model: YOLO11n-P2 (4 detection scales: P2, P3, P4, P5)")
    print(f"Resolution: 640 x 640")
    print(f"Batch Size: 2")
    print(f"Epochs: 1 (STRICT 1-epoch feasibility constraint)")
    print(f"Device: CPU (workers=0, seed=42)")
    print(f"Output: {EXP_DIR}")
    
    start_time = time.time()
    
    results = model.train(
        data=str(DATA_YAML),
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        workers=0,
        seed=42,
        deterministic=True,
        project=str(EXPERIMENTS_DIR),
        name=EXP_NAME,
        exist_ok=True,
        val=True,
        save=True,
        plots=True,
        verbose=True
    )
    
    elapsed_time = time.time() - start_time
    mem_after = psutil.virtual_memory()
    peak_rss = process.memory_info().rss / (1024**2)
    
    print(f"\n=== Feasibility Preflight Complete ===")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Peak Process RSS: {peak_rss:.2f} MB")
    print(f"Final Host Available RAM: {mem_after.available / (1024**3):.2f} GB")
    
    # Check outputs
    weights_dir = EXP_DIR / "weights"
    last_pt = weights_dir / "last.pt"
    results_csv = EXP_DIR / "results.csv"
    
    assert last_pt.exists(), f"Expected checkpoint missing at {last_pt}"
    ckpt_size_mb = round(last_pt.stat().st_size / (1024**2), 2)
    
    epoch_metrics = {}
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split(",")]
                vals = [v.strip() for v in lines[1].split(",")]
                epoch_metrics = dict(zip(headers, vals))
    
    summary = {
        "architecture": "YOLO11n-P2",
        "resolution": 640,
        "batch_size": 2,
        "epochs_completed": 1,
        "elapsed_seconds": round(elapsed_time, 2),
        "elapsed_minutes": round(elapsed_time / 60, 2),
        "peak_process_rss_mb": round(peak_rss, 2),
        "initial_available_ram_gb": round(mem_before.available / (1024**3), 2),
        "final_available_ram_gb": round(mem_after.available / (1024**3), 2),
        "checkpoint_path": str(last_pt),
        "checkpoint_size_mb": ckpt_size_mb,
        "epoch_1_metrics": epoch_metrics,
        "hardware": "12th Gen Intel Core i7-12650H (10 cores / 16 threads, CPU)"
    }
    
    summary_file = EXP_DIR / "p2_preflight_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Preflight summary recorded at {summary_file}")

if __name__ == "__main__":
    main()
