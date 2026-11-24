r"""EcoSort AI — Full YOLO11n-P2 13-Class Waste Detection Training and Diagnostic Evaluation.
Task 5.4.3: 50 epochs, batch=2, imgsz=640, CPU, seed=42, patience=15.
Evaluates strictly on validation split. Holdout test split is LOCKED and UNACCESSED.
"""

import time
import os
import psutil
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# Paths
INITIAL_CKPT = Path(r"D:\EcoSort-Resources\experiments\yolo11n_p2_architecture_preflight\yolo11n_p2_initial.pt")
YAML_PATH = Path(r"ecosort-ai\models\yolo11n-p2.yaml")
DATA_YAML = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\data.yaml")
EXPERIMENTS_DIR = Path(r"D:\EcoSort-Resources\experiments")
EXP_NAME = "yolo11n_p2_baseline_13class"
EXP_DIR = EXPERIMENTS_DIR / EXP_NAME
VAL_IMAGES_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\images\val")
VAL_LABELS_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\labels\val")

BASELINE_640_DIR = EXPERIMENTS_DIR / "yolo11n_baseline_13class"
EXP_1280_DIR = EXPERIMENTS_DIR / "yolo11n_1280_baseline"

CLASS_NAMES = [
    "clear_plastic_bottle",
    "glass_bottle",
    "plastic_bottle_cap",
    "metal_bottle_cap",
    "drink_can",
    "other_carton",
    "corrugated_carton",
    "disposable_plastic_cup",
    "other_plastic",
    "plastic_film",
    "other_plastic_wrapper",
    "styrofoam_piece",
    "cigarette"
]

def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def run_pre_training_checks():
    print("=== Step 1: Pre-Training Verification & Safety Checks ===")
    assert INITIAL_CKPT.exists(), f"Initial checkpoint missing: {INITIAL_CKPT}"
    assert DATA_YAML.exists(), f"Dataset config missing: {DATA_YAML}"
    assert YAML_PATH.exists(), f"Architecture YAML missing: {YAML_PATH}"

    # Load model and verify architecture
    model = YOLO(str(INITIAL_CKPT))
    detect_module = model.model.model[-1]
    strides = [float(s) for s in detect_module.stride]
    nl = int(detect_module.nl)
    nc = int(detect_module.nc)
    total_params = sum(p.numel() for p in model.model.parameters())

    print(f"Checkpoint loaded: {INITIAL_CKPT}")
    print(f"Detection Strides: {strides} (Expected: [4.0, 8.0, 16.0, 32.0])")
    assert strides == [4.0, 8.0, 16.0, 32.0], f"Stride mismatch: {strides}"
    print(f"Detection Scales (nl): {nl} (Expected: 4)")
    assert nl == 4, f"Scale count mismatch: {nl}"
    print(f"Classes (nc): {nc} (Expected: 13)")
    assert nc == 13, f"Class count mismatch: {nc}"
    print(f"Total Parameters: {total_params:,}")

    # Check host resources
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("D:\\")
    print(f"Host Available RAM: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")
    print(f"D: Drive Space: {disk.free / (1024**3):.2f} GB free")
    assert mem.available / (1024**3) >= 2.0, "Insufficient RAM (< 2.0 GB)"
    assert disk.free / (1024**3) >= 10.0, "Insufficient disk space (< 10 GB)"

    # Verify dataset labels
    val_labels = list(VAL_LABELS_DIR.glob("*.txt"))
    assert len(val_labels) > 0, "No validation labels found"
    max_cid = -1
    for p in val_labels:
        for line in open(p):
            parts = line.strip().split()
            if parts:
                cid = int(parts[0])
                if cid > max_cid: max_cid = cid
    print(f"Validation label files: {len(val_labels)}, Max class ID: {max_cid} (Expected <= 12)")
    assert max_cid < 13, f"Label ID {max_cid} out of range for 13 classes"

    print("Pre-training safety checks PASSED.\n")
    return model, mem.available, disk.free

def train_p2_model():
    model, initial_ram, initial_disk = run_pre_training_checks()

    print("=== Step 2: Launching Full YOLO11n-P2 50-Epoch Training ===")
    print(f"Output Directory: {EXP_DIR}")
    print(f"Epochs: 50 max (Patience=15 early stopping)")
    print(f"Batch Size: 2")
    print(f"Resolution: 640 x 640")
    print(f"Device: CPU (workers=0, seed=42)")
    print(f"Optimizer: SGD (lr0=0.01, momentum=0.937, weight_decay=0.0005, close_mosaic=10)")

    process = psutil.Process()
    start_time = time.time()

    # Train model starting from the initial transferred checkpoint
    train_results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        batch=2,
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
        verbose=True
    )

    elapsed_time = time.time() - start_time
    mem_after = psutil.virtual_memory()
    peak_rss = process.memory_info().rss / (1024**2)

    print("\n=== Training Completed ===")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes / {elapsed_time/3600:.2f} hours)")
    print(f"Peak Process RSS: {peak_rss:.2f} MB")
    print(f"Final Host Available RAM: {mem_after.available / (1024**3):.2f} GB")

    best_pt = EXP_DIR / "weights" / "best.pt"
    last_pt = EXP_DIR / "weights" / "last.pt"
    assert best_pt.exists(), f"best.pt missing at {best_pt}"
    assert last_pt.exists(), f"last.pt missing at {last_pt}"

    # Step 3: Run comprehensive validation diagnostics
    print("\n=== Step 3: Running Validation Diagnostics on YOLO11n-P2 Best Model ===")
    diag_summary = run_validation_diagnostics(best_pt, elapsed_time, peak_rss, initial_ram, mem_after.available)

    # Step 4: Run 3-architecture comparative benchmark
    print("\n=== Step 4: Running Three-Architecture Comparative Analysis ===")
    compare_summary = run_three_architecture_comparison(best_pt, diag_summary)

    print("\n=== All Steps Completed Successfully ===")
    return diag_summary, compare_summary

def run_validation_diagnostics(best_weights_path, elapsed_time, peak_rss, initial_ram, final_ram):
    results_csv = EXP_DIR / "results.csv"
    epochs_data = []
    best_epoch_idx = 1
    best_fitness = 0.0

    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split(",")]
                for line in lines[1:]:
                    vals = [v.strip() for v in line.split(",")]
                    row = dict(zip(headers, vals))
                    epochs_data.append(row)
                    map50 = float(row.get("metrics/mAP50(B)", 0))
                    map50_95 = float(row.get("metrics/mAP50-95(B)", 0))
                    fitness = 0.1 * map50 + 0.9 * map50_95
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_epoch_idx = int(row.get("epoch", 1))

    best_epoch_row = epochs_data[best_epoch_idx - 1] if epochs_data else {}

    # Official Ultralytics Validation on best checkpoint
    print("Running official Ultralytics validation on best.pt at imgsz=640...")
    best_model = YOLO(str(best_weights_path))
    val_eval = best_model.val(data=str(DATA_YAML), imgsz=640, split="val", device="cpu", verbose=True)

    official_metrics = {}
    if hasattr(val_eval, "results_dict"):
        for k, v in val_eval.results_dict.items():
            try: official_metrics[k] = float(v)
            except Exception: official_metrics[k] = str(v)

    per_class_official = {}
    names = best_model.names
    for i in range(len(names)):
        cname = names[i]
        p_map50 = float(val_eval.box.maps[i]) if hasattr(val_eval.box, "maps") else 0.0
        per_class_official[cname] = {
            "class_id": i,
            "mAP50_95": round(p_map50, 4)
        }

    # Diagnostic Scale-Specific Evaluation at conf=0.25 and conf=0.05
    val_img_paths = sorted(list(VAL_IMAGES_DIR.glob("*.jpg")))
    scale_diagnostics = {}

    for conf_thresh in [0.25, 0.05]:
        preds = best_model.predict(source=[str(p) for p in val_img_paths], conf=conf_thresh, iou=0.45, device="cpu", imgsz=640, verbose=False)
        scale_stats = {
            "small": {"total": 0, "detected": 0},
            "medium": {"total": 0, "detected": 0},
            "large": {"total": 0, "detected": 0}
        }
        per_class_diag = {c: {"gt": 0, "tp": 0, "fn": 0} for c in CLASS_NAMES}

        for res in preds:
            stem = Path(res.path).stem
            orig_h, orig_w = res.orig_shape
            lbl_file = VAL_LABELS_DIR / f"{stem}.txt"
            gts = []
            if lbl_file.exists():
                with open(lbl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts: continue
                        cid = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        area_640 = (w * 640) * (h * 640)
                        scale = "small" if area_640 < 1024 else ("medium" if area_640 <= 9216 else "large")
                        x1 = (xc - w / 2) * orig_w
                        y1 = (yc - h / 2) * orig_h
                        x2 = (xc + w / 2) * orig_w
                        y2 = (yc + h / 2) * orig_h
                        gts.append({"cid": cid, "cname": CLASS_NAMES[cid], "box": [x1, y1, x2, y2], "scale": scale, "matched": False})
                        scale_stats[scale]["total"] += 1
                        per_class_diag[CLASS_NAMES[cid]]["gt"] += 1

            pred_boxes = res.boxes
            if len(pred_boxes) > 0:
                for pb, pc, pconf in zip(pred_boxes.xyxy.cpu().numpy(), pred_boxes.cls.cpu().numpy().astype(int), pred_boxes.conf.cpu().numpy()):
                    best_iou = 0.0
                    best_idx = -1
                    for idx, g in enumerate(gts):
                        iou = box_iou(pb, g["box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                    if best_iou >= 0.5 and pc == gts[best_idx]["cid"]:
                        if not gts[best_idx]["matched"]:
                            gts[best_idx]["matched"] = True
                            scale_stats[gts[best_idx]["scale"]]["detected"] += 1
                            per_class_diag[CLASS_NAMES[pc]]["tp"] += 1

        for sc in scale_stats:
            tot = scale_stats[sc]["total"]
            det = scale_stats[sc]["detected"]
            scale_stats[sc]["recall"] = round(det / tot if tot > 0 else 0.0, 4)

        for c in per_class_diag:
            gt = per_class_diag[c]["gt"]
            tp = per_class_diag[c]["tp"]
            per_class_diag[c]["fn"] = gt - tp
            per_class_diag[c]["recall"] = round(tp / gt if gt > 0 else 0.0, 4)

        conf_key = f"conf_{str(conf_thresh).replace('.', '')}"
        scale_diagnostics[conf_key] = {
            "scales": scale_stats,
            "per_class": per_class_diag
        }

    # Generate Visualization Canvases
    vis_dir = EXP_DIR / "error_analysis" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    res_025 = best_model.predict(source=[str(p) for p in val_img_paths], conf=0.25, iou=0.45, device="cpu", imgsz=640, verbose=False)
    selected_indices = [0, 4, 8, 12, 18, 25]

    for idx in selected_indices:
        if idx >= len(val_img_paths): continue
        ip = val_img_paths[idx]
        stem = ip.stem
        res = res_025[idx]
        orig_h, orig_w = res.orig_shape

        img = cv2.imread(str(ip))
        if img is None: continue
        left = img.copy()
        right = img.copy()

        lbl_file = VAL_LABELS_DIR / f"{stem}.txt"
        gts = []
        if lbl_file.exists():
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    x1 = int((xc - w / 2) * orig_w)
                    y1 = int((yc - h / 2) * orig_h)
                    x2 = int((xc + w / 2) * orig_w)
                    y2 = int((yc + h / 2) * orig_h)
                    cname = CLASS_NAMES[cid]
                    gts.append({"cid": cid, "cname": cname, "box": [x1, y1, x2, y2]})
                    cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(left, f"GT: {cname}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        pred_boxes = res.boxes
        if len(pred_boxes) > 0:
            for pb, pc, pconf in zip(pred_boxes.xyxy.cpu().numpy(), pred_boxes.cls.cpu().numpy().astype(int), pred_boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, pb)
                pname = CLASS_NAMES[pc]
                best_iou = 0.0
                match_class = -1
                for g in gts:
                    iou = box_iou(pb, g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        match_class = g["cid"]

                if best_iou >= 0.5 and pc == match_class:
                    color = (0, 255, 0)
                    tag = f"TP: {pname} {pconf:.2f}"
                elif best_iou >= 0.5 and pc != match_class:
                    color = (255, 0, 255)
                    tag = f"CONFUSED: {pname} {pconf:.2f}"
                else:
                    color = (255, 200, 0)
                    tag = f"FP: {pname} {pconf:.2f}"

                cv2.rectangle(right, (x1, y1), (x2, y2), color, 2)
                cv2.putText(right, tag, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(left, f"GT ({stem})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(right, f"P2 Pred ({stem})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        h_disp = 640
        w_disp = int(orig_w * (640 / orig_h))
        left_r = cv2.resize(left, (w_disp, h_disp))
        right_r = cv2.resize(right, (w_disp, h_disp))
        canvas = np.hstack([left_r, right_r])

        out_vis = vis_dir / f"val_p2_comparison_{stem}.jpg"
        cv2.imwrite(str(out_vis), canvas)

    summary = {
        "experiment_name": EXP_NAME,
        "model_architecture": "YOLO11n-P2",
        "resolution": 640,
        "batch_size": 2,
        "epochs_max": 50,
        "completed_epochs": len(epochs_data),
        "early_stopped": len(epochs_data) < 50,
        "best_epoch": best_epoch_idx,
        "training_duration_seconds": round(elapsed_time, 2),
        "training_duration_minutes": round(elapsed_time / 60, 2),
        "training_duration_hours": round(elapsed_time / 3600, 2),
        "peak_process_rss_mb": round(peak_rss, 2),
        "initial_available_ram_gb": round(initial_ram / (1024**3), 2),
        "final_available_ram_gb": round(final_ram / (1024**3), 2),
        "best_epoch_metrics": best_epoch_row,
        "official_val_metrics": official_metrics,
        "per_class_official": per_class_official,
        "scale_diagnostics": scale_diagnostics,
        "per_class_validation": scale_diagnostics["conf_025"]["per_class"]
    }

    out_file = EXP_DIR / "training_and_validation_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {out_file}")
    return summary

def run_three_architecture_comparison(p2_best_weights, p2_summary):
    # Benchmark CPU Latency on Best P2 Weights
    p2_model = YOLO(str(p2_best_weights))
    dummy_input = torch.zeros(1, 3, 640, 640)
    for _ in range(10): _ = p2_model(dummy_input, verbose=False)
    p2_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = p2_model(dummy_input, verbose=False)
        p2_times.append((time.perf_counter() - t0) * 1000)
    p2_mean_ms = round(float(np.mean(p2_times)), 2)
    p2_fps = round(float(1000.0 / p2_mean_ms), 1)

    # Load 640 baseline summary
    base_summary_file = BASELINE_640_DIR / "baseline_training_summary.json"
    base_summary = {}
    if base_summary_file.exists():
        with open(base_summary_file, "r", encoding="utf-8") as f:
            base_summary = json.load(f)

    # Load 1280 experiment summary
    exp1280_summary_file = EXP_1280_DIR / "training_and_validation_summary.json"
    exp1280_summary = {}
    if exp1280_summary_file.exists():
        with open(exp1280_summary_file, "r", encoding="utf-8") as f:
            exp1280_summary = json.load(f)

    # Load official resolution comparison
    res_comp_file = EXP_1280_DIR / "resolution_comparison_official.json"
    official_comp = {}
    if res_comp_file.exists():
        with open(res_comp_file, "r", encoding="utf-8") as f:
            official_comp = json.load(f)

    comparison = {
        "model_640_baseline": {
            "name": "YOLO11n @ 640",
            "resolution": 640,
            "scales": 3,
            "strides": [8, 16, 32],
            "spatial_anchors": 8400,
            "parameters": 2607799,
            "checkpoint_size_mb": 5.26,
            "latency_ms": 39.3,
            "fps": 25.4,
            "peak_rss_mb": 523.29,
            "training_duration_mins": 77.18,
            "best_epoch": 42,
            "val_precision": float(official_comp.get("model_640", {}).get("all", {}).get("p", 0.3207)),
            "val_recall": float(official_comp.get("model_640", {}).get("all", {}).get("r", 0.2637)),
            "val_map50": float(official_comp.get("model_640", {}).get("all", {}).get("map50", 0.1471)),
            "val_map50_95": float(official_comp.get("model_640", {}).get("all", {}).get("map50_95", 0.1084)),
            "small_recall_conf025": 0.0,
            "small_recall_conf005": 0.0238
        },
        "model_1280_experiment": {
            "name": "YOLO11n @ 1280",
            "resolution": 1280,
            "scales": 3,
            "strides": [8, 16, 32],
            "spatial_anchors": 33600,
            "parameters": 2607799,
            "checkpoint_size_mb": 5.41,
            "latency_ms": 163.81,
            "fps": 6.1,
            "peak_rss_mb": 650.93,
            "training_duration_mins": 514.09,
            "best_epoch": 32,
            "val_precision": float(official_comp.get("model_1280", {}).get("all", {}).get("p", 0.4296)),
            "val_recall": float(official_comp.get("model_1280", {}).get("all", {}).get("r", 0.1692)),
            "val_map50": float(official_comp.get("model_1280", {}).get("all", {}).get("map50", 0.1913)),
            "val_map50_95": float(official_comp.get("model_1280", {}).get("all", {}).get("map50_95", 0.1393)),
            "small_recall_conf025": 0.0,
            "small_recall_conf005": 0.0476
        },
        "model_p2_experiment": {
            "name": "YOLO11n-P2 @ 640",
            "resolution": 640,
            "scales": 4,
            "strides": [4, 8, 16, 32],
            "spatial_anchors": 34000,
            "parameters": 2668404,
            "checkpoint_size_mb": round(p2_best_weights.stat().st_size / (1024**2), 2),
            "latency_ms": p2_mean_ms,
            "fps": p2_fps,
            "peak_rss_mb": p2_summary.get("peak_process_rss_mb", 0.0),
            "training_duration_mins": p2_summary.get("training_duration_minutes", 0.0),
            "best_epoch": p2_summary.get("best_epoch", 1),
            "val_precision": float(p2_summary.get("official_val_metrics", {}).get("metrics/precision(B)", 0.0)),
            "val_recall": float(p2_summary.get("official_val_metrics", {}).get("metrics/recall(B)", 0.0)),
            "val_map50": float(p2_summary.get("official_val_metrics", {}).get("metrics/mAP50(B)", 0.0)),
            "val_map50_95": float(p2_summary.get("official_val_metrics", {}).get("metrics/mAP50-95(B)", 0.0)),
            "small_recall_conf025": p2_summary.get("scale_diagnostics", {}).get("conf_025", {}).get("scales", {}).get("small", {}).get("recall", 0.0),
            "small_recall_conf005": p2_summary.get("scale_diagnostics", {}).get("conf_005", {}).get("scales", {}).get("small", {}).get("recall", 0.0)
        }
    }

    comp_file = EXP_DIR / "three_architecture_comparison.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"Three-architecture comparison written to {comp_file}")
    return comparison

if __name__ == "__main__":
    train_p2_model()
