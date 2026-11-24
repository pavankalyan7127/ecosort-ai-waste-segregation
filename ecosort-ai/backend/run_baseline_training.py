r"""Script to execute YOLO11n baseline training (50 epochs max) on 13 candidate Dry waste classes,
followed by an unbiased evaluation on the untouched test split.
"""

import time
import os
import psutil
import json
import shutil
from pathlib import Path
from ultralytics import YOLO

# Configuration
PRETRAINED_WEIGHTS = Path(r"D:\EcoSort-Resources\models\pretrained\yolo11n.pt")
DATA_YAML = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\data.yaml")
EXPERIMENTS_DIR = Path(r"D:\EcoSort-Resources\experiments")
EXP_NAME = "yolo11n_baseline_13class"
EXP_DIR = EXPERIMENTS_DIR / EXP_NAME

def pre_training_checks():
    print("=== Step 1: Pre-Training Environment and Safety Checks ===")
    assert PRETRAINED_WEIGHTS.exists(), f"Pretrained weights missing: {PRETRAINED_WEIGHTS}"
    assert DATA_YAML.exists(), f"Dataset config missing: {DATA_YAML}"

    # Check disk space on D:
    disk = psutil.disk_usage("D:\\")
    free_gb = disk.free / (1024**3)
    print(f"Disk Space on D:\\: {free_gb:.2f} GB free of {disk.total / (1024**3):.2f} GB total")
    assert free_gb >= 2.0, f"Insufficient disk space on D: drive ({free_gb:.2f} GB)"

    # Check RAM
    mem = psutil.virtual_memory()
    avail_ram_gb = mem.available / (1024**3)
    total_ram_gb = mem.total / (1024**3)
    print(f"System RAM: {avail_ram_gb:.2f} GB available of {total_ram_gb:.2f} GB total")
    assert avail_ram_gb >= 1.5, f"Insufficient available RAM: {avail_ram_gb:.2f} GB"

    # Verify no accidental overwrite of preflight test
    preflight_dir = EXPERIMENTS_DIR / "yolo11n_preflight_test"
    assert preflight_dir.exists(), "Preflight directory missing; cannot verify safety"

    print("Pre-training safety checks passed successfully.\n")
    return {
        "free_disk_gb": round(free_gb, 2),
        "initial_available_ram_gb": round(avail_ram_gb, 2),
        "total_ram_gb": round(total_ram_gb, 2),
    }

def train_and_evaluate():
    env_info = pre_training_checks()

    print("=== Step 2: Initializing Model and Training Pipeline ===")
    print(f"Base checkpoint: {PRETRAINED_WEIGHTS} (Clean COCO Pretrained)")
    print(f"Experiment Directory: {EXP_DIR}")
    model = YOLO(str(PRETRAINED_WEIGHTS))

    start_train_time = time.time()
    process = psutil.Process()
    initial_rss = process.memory_info().rss / (1024**2)

    print("Starting YOLO11n fine-tuning for up to 50 epochs with early stopping patience=15...")
    train_results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        batch=4,
        imgsz=640,
        device="cpu",
        workers=0,
        seed=42,
        deterministic=True,
        patience=15,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        close_mosaic=10,
        project=str(EXPERIMENTS_DIR),
        name=EXP_NAME,
        exist_ok=True,
        val=True,
        save=True,
        plots=True,
        verbose=True,
    )

    train_duration = time.time() - start_train_time
    peak_rss = process.memory_info().rss / (1024**2)
    final_mem = psutil.virtual_memory()

    print(f"\nTraining completed in {train_duration:.2f} seconds ({train_duration/60:.2f} minutes).")
    print(f"Peak Process RSS: {peak_rss:.2f} MB")
    print(f"Remaining Available RAM: {final_mem.available / (1024**3):.2f} GB")

    best_checkpoint = EXP_DIR / "weights" / "best.pt"
    last_checkpoint = EXP_DIR / "weights" / "last.pt"
    assert best_checkpoint.exists(), f"Best checkpoint not found at {best_checkpoint}"
    print(f"Verified best checkpoint: {best_checkpoint} ({best_checkpoint.stat().st_size / (1024**2):.2f} MB)")

    # Read training CSV for epoch count and loss curves
    results_csv = EXP_DIR / "results.csv"
    epochs_data = []
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split(",")]
                for line in lines[1:]:
                    vals = [v.strip() for v in line.split(",")]
                    epochs_data.append(dict(zip(headers, vals)))

    completed_epochs = len(epochs_data)
    print(f"Actual completed epochs: {completed_epochs}")

    # Step 3: Final Holdout Test Set Evaluation (EXACTLY ONCE)
    print("\n=== Step 3: Evaluating Best Checkpoint on Untouched Test Split ===")
    test_model = YOLO(str(best_checkpoint))

    start_test_time = time.time()
    test_metrics = test_model.val(
        data=str(DATA_YAML),
        split="test",
        batch=4,
        imgsz=640,
        device="cpu",
        workers=0,
        project=str(EXP_DIR),
        name="test_evaluation",
        exist_ok=True,
        plots=True,
        save_json=True,
    )
    test_duration = time.time() - start_test_time
    print(f"Test evaluation completed in {test_duration:.2f} seconds.")

    # Parse test metrics
    overall_metrics = {
        "precision": float(test_metrics.box.mp),
        "recall": float(test_metrics.box.mr),
        "mAP50": float(test_metrics.box.map50),
        "mAP50_95": float(test_metrics.box.map),
    }
    print(f"\nFinal Test Set Overall Metrics:")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall:    {overall_metrics['recall']:.4f}")
    print(f"  mAP@50:    {overall_metrics['mAP50']:.4f}")
    print(f"  mAP@50-95: {overall_metrics['mAP50_95']:.4f}")

    # Per-class test metrics
    class_names = test_metrics.names
    per_class_results = {}
    print("\nPer-Class Holdout Test Metrics:")
    for i, name in class_names.items():
        ap50 = float(test_metrics.box.ap50[i]) if i < len(test_metrics.box.ap50) else 0.0
        ap50_95 = float(test_metrics.box.ap[i]) if i < len(test_metrics.box.ap) else 0.0
        p = float(test_metrics.box.p[i]) if i < len(test_metrics.box.p) else 0.0
        r = float(test_metrics.box.r[i]) if i < len(test_metrics.box.r) else 0.0

        per_class_results[name] = {
            "class_id": i,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "ap50": round(ap50, 4),
            "ap50_95": round(ap50_95, 4),
        }
        print(f"  [{i:2d}] {name:<25}: AP50={ap50:.4f}, AP50-95={ap50_95:.4f}, P={p:.4f}, R={r:.4f}")

    # Save comprehensive summary JSON
    summary_payload = {
        "experiment_name": EXP_NAME,
        "model_architecture": "YOLO11n",
        "dataset_config": str(DATA_YAML),
        "base_pretrained_weights": str(PRETRAINED_WEIGHTS),
        "hyperparameters": {
            "epochs_max": 50,
            "actual_completed_epochs": completed_epochs,
            "batch_size": 4,
            "image_size": 640,
            "device": "cpu",
            "workers": 0,
            "seed": 42,
            "patience": 15,
            "optimizer": "SGD",
            "lr0": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "close_mosaic": 10,
        },
        "system_and_environment": {
            "cpu": "12th Gen Intel Core i7-12650H (10 cores / 16 threads)",
            "initial_env": env_info,
            "training_duration_seconds": round(train_duration, 2),
            "training_duration_minutes": round(train_duration / 60, 2),
            "test_duration_seconds": round(test_duration, 2),
            "peak_process_rss_mb": round(peak_rss, 2),
            "final_available_ram_gb": round(final_mem.available / (1024**3), 2),
        },
        "checkpoints": {
            "best": str(best_checkpoint),
            "last": str(last_checkpoint),
            "best_size_mb": round(best_checkpoint.stat().st_size / (1024**2), 2),
        },
        "validation_progression": epochs_data,
        "final_holdout_test_metrics": {
            "overall": overall_metrics,
            "per_class": per_class_results,
        }
    }

    summary_file = EXP_DIR / "baseline_training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\nAll artifacts and summary recorded at {summary_file}")
    print("Baseline training and final evaluation completed successfully.")

if __name__ == "__main__":
    train_and_evaluate()
