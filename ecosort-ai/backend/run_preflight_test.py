r"""Script to run 1-epoch YOLO11n preflight training feasibility test.
"""

import time
import os
import psutil
import json
from pathlib import Path
from ultralytics import YOLO

MODEL_WEIGHTS = Path(r"D:\EcoSort-Resources\models\pretrained\yolo11n.pt")
DATA_YAML = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\data.yaml")
EXPERIMENTS_DIR = Path(r"D:\EcoSort-Resources\experiments")
EXP_NAME = "yolo11n_preflight_test"

def run_preflight():
    print("=== Starting 1-Epoch Preflight Feasibility Test ===")
    process = psutil.Process()
    ram_before = psutil.virtual_memory()
    print(f"Initial System RAM Available: {ram_before.available / (1024**3):.2f} GB / {ram_before.total / (1024**3):.2f} GB")
    print(f"Initial Process RSS: {process.memory_info().rss / (1024**2):.2f} MB")

    # Load model
    print(f"Loading pretrained model: {MODEL_WEIGHTS}")
    model = YOLO(str(MODEL_WEIGHTS))

    start_time = time.time()
    cpu_percent_start = psutil.cpu_percent(interval=1.0)

    # Train for exactly 1 epoch with batch=4, imgsz=640, workers=0 (conservative Windows setting)
    results = model.train(
        data=str(DATA_YAML),
        epochs=1,
        batch=4,
        imgsz=640,
        device="cpu",
        workers=0,
        project=str(EXPERIMENTS_DIR),
        name=EXP_NAME,
        exist_ok=True,
        seed=42,
        deterministic=True,
        verbose=True,
        plots=True,
        save=True,
        val=True,
    )

    elapsed_time = time.time() - start_time
    ram_after = psutil.virtual_memory()
    process_mem = process.memory_info().rss / (1024**2)

    print("\n=== Preflight Run Completed Successfully ===")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Process Peak RSS: {process_mem:.2f} MB")
    print(f"Final System RAM Available: {ram_after.available / (1024**3):.2f} GB")

    # Inspect results and save preflight summary
    exp_path = EXPERIMENTS_DIR / EXP_NAME
    results_csv = exp_path / "results.csv"

    metrics_data = {}
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split(",")]
                values = [v.strip() for v in lines[-1].split(",")]
                metrics_data = dict(zip(headers, values))
                print("\nMetrics from results.csv:")
                for k, v in metrics_data.items():
                    print(f"  {k}: {v}")

    summary_data = {
        "status": "SUCCESS",
        "epochs": 1,
        "batch_size": 4,
        "image_size": 640,
        "device": "cpu",
        "workers": 0,
        "seed": 42,
        "elapsed_seconds": round(elapsed_time, 2),
        "initial_available_ram_gb": round(ram_before.available / (1024**3), 2),
        "final_available_ram_gb": round(ram_after.available / (1024**3), 2),
        "process_rss_mb": round(process_mem, 2),
        "experiment_dir": str(exp_path),
        "metrics": metrics_data,
        "model_checkpoint_created": (exp_path / "weights" / "best.pt").exists() or (exp_path / "weights" / "last.pt").exists(),
    }

    out_file = exp_path / "preflight_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved preflight summary to {out_file}")

if __name__ == "__main__":
    run_preflight()
