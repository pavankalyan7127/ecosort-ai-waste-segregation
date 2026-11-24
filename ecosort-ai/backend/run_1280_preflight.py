r"""Script to run 1-epoch feasibility preflights at 1280x1280 resolution (batch=1 and batch=2).
"""

import time
import psutil
import json
from pathlib import Path
from ultralytics import YOLO

PRETRAINED_WEIGHTS = Path(r"D:\EcoSort-Resources\models\pretrained\yolo11n.pt")
DATA_YAML = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\data.yaml")
EXPERIMENTS_DIR = Path(r"D:\EcoSort-Resources\experiments")
PREFLIGHT_DIR_B1 = EXPERIMENTS_DIR / "yolo11n_1280_preflight_b1"
PREFLIGHT_DIR_B2 = EXPERIMENTS_DIR / "yolo11n_1280_preflight_b2"

def test_epoch(batch_size, exp_dir):
    print(f"\n=======================================================")
    print(f"Testing 1-Epoch Preflight at imgsz=1280, batch={batch_size}")
    print(f"Output Directory: {exp_dir}")
    print(f"=======================================================")

    process = psutil.Process()
    ram_before = psutil.virtual_memory()
    initial_rss = process.memory_info().rss / (1024**2)

    print(f"Initial Host Available RAM: {ram_before.available / (1024**3):.2f} GB")
    print(f"Initial Process RSS: {initial_rss:.2f} MB")

    start_time = time.time()
    model = YOLO(str(PRETRAINED_WEIGHTS))

    results = model.train(
        data=str(DATA_YAML),
        epochs=1,
        batch=batch_size,
        imgsz=1280,
        device="cpu",
        workers=0,
        project=str(EXPERIMENTS_DIR),
        name=exp_dir.name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        verbose=True,
        plots=True,
        save=True,
        val=True,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
    )

    elapsed_time = time.time() - start_time
    ram_after = psutil.virtual_memory()
    peak_rss = process.memory_info().rss / (1024**2)

    print(f"\n--- Preflight batch={batch_size} Completed Successfully ---")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Peak Process RSS: {peak_rss:.2f} MB")
    print(f"Final Host Available RAM: {ram_after.available / (1024**3):.2f} GB")

    best_pt = exp_dir / "weights" / "best.pt"
    assert best_pt.exists(), f"Checkpoint best.pt missing at {best_pt}"

    summary = {
        "batch_size": batch_size,
        "image_size": 1280,
        "device": "cpu",
        "workers": 0,
        "elapsed_seconds": round(elapsed_time, 2),
        "elapsed_minutes": round(elapsed_time / 60, 2),
        "initial_available_ram_gb": round(ram_before.available / (1024**3), 2),
        "final_available_ram_gb": round(ram_after.available / (1024**3), 2),
        "peak_process_rss_mb": round(peak_rss, 2),
        "checkpoint_size_mb": round(best_pt.stat().st_size / (1024**2), 2),
    }

    with open(exp_dir / "preflight_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

def main():
    print("=== Step 1: Preflight Test at Batch Size 1 ===")
    res_b1 = test_epoch(batch_size=1, exp_dir=PREFLIGHT_DIR_B1)

    print("\n=== Step 2: Investigating Batch Size 2 ===")
    res_b2 = test_epoch(batch_size=2, exp_dir=PREFLIGHT_DIR_B2)

    print("\n================ PREFLIGHT COMPARISON SUMMARY ================")
    print(f"Batch 1: Time={res_b1['elapsed_seconds']}s ({res_b1['elapsed_minutes']}m), Peak RSS={res_b1['peak_process_rss_mb']}MB, Remaining RAM={res_b1['final_available_ram_gb']}GB")
    print(f"Batch 2: Time={res_b2['elapsed_seconds']}s ({res_b2['elapsed_minutes']}m), Peak RSS={res_b2['peak_process_rss_mb']}MB, Remaining RAM={res_b2['final_available_ram_gb']}GB")

    # Save overall comparison
    with open(EXPERIMENTS_DIR / "preflight_1280_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"batch_1": res_b1, "batch_2": res_b2}, f, indent=2)

if __name__ == "__main__":
    main()
