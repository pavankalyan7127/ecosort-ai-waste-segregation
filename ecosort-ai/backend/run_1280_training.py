r"""Script to execute YOLO11n 1280x1280 training on 13 candidate classes,
followed by comprehensive validation diagnostics and per-scale comparison against 640 baseline.
STRICT RULE: Test split is NOT evaluated. Only validation split is used.
"""

import time
import os
import psutil
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import cv2
from ultralytics import YOLO

PRETRAINED_WEIGHTS = Path(r"D:\EcoSort-Resources\models\pretrained\yolo11n.pt")
DATA_YAML = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\data.yaml")
EXPERIMENTS_DIR = Path(r"D:\EcoSort-Resources\experiments")
EXP_NAME = "yolo11n_1280_baseline"
EXP_DIR = EXPERIMENTS_DIR / EXP_NAME
VAL_IMAGES_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\images\val")
VAL_LABELS_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\labels\val")

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
    "cigarette",
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

def xywhn2xyxy(xc, yc, w, h, img_w=1280, img_h=1280):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]

def train_1280():
    print("=== Step 1: Pre-Training Verification ===")
    assert PRETRAINED_WEIGHTS.exists(), f"Missing weights: {PRETRAINED_WEIGHTS}"
    assert DATA_YAML.exists(), f"Missing data.yaml: {DATA_YAML}"

    mem = psutil.virtual_memory()
    print(f"Host Available RAM: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")
    disk = psutil.disk_usage("D:\\")
    print(f"D: Drive Space: {disk.free / (1024**3):.2f} GB free")

    process = psutil.Process()
    initial_rss = process.memory_info().rss / (1024**2)

    print("\n=== Step 2: Launching YOLO11n Training at 1280x1280 ===")
    print(f"Model: YOLO11n (Pretrained COCO)")
    print(f"Resolution: 1280 x 1280")
    print(f"Batch Size: 2")
    print(f"Max Epochs: 50 (Patience=15)")
    print(f"Output: {EXP_DIR}")

    model = YOLO(str(PRETRAINED_WEIGHTS))
    start_time = time.time()

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        batch=2,
        imgsz=1280,
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

    elapsed_time = time.time() - start_time
    ram_after = psutil.virtual_memory()
    peak_rss = process.memory_info().rss / (1024**2)

    print(f"\n=== Training Finished ===")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes / {elapsed_time/3600:.2f} hours)")
    print(f"Peak Process RSS: {peak_rss:.2f} MB")
    print(f"Final Host Available RAM: {ram_after.available / (1024**3):.2f} GB")

    best_pt = EXP_DIR / "weights" / "best.pt"
    last_pt = EXP_DIR / "weights" / "last.pt"
    assert best_pt.exists(), f"Missing best.pt at {best_pt}"

    # Step 3: Run Validation Diagnostics using best.pt
    print("\n=== Step 3: Running Validation Diagnostics on 1280 Best Model ===")
    diag_summary = run_validation_diagnostics(best_pt)

    # Save comprehensive summary
    full_summary = {
        "experiment_name": EXP_NAME,
        "resolution": 1280,
        "batch_size": 2,
        "epochs_max": 50,
        "actual_completed_epochs": diag_summary["completed_epochs"],
        "early_stopped": diag_summary["completed_epochs"] < 50,
        "best_epoch": diag_summary["best_epoch"],
        "training_duration_seconds": round(elapsed_time, 2),
        "training_duration_minutes": round(elapsed_time / 60, 2),
        "training_duration_hours": round(elapsed_time / 3600, 2),
        "peak_process_rss_mb": round(peak_rss, 2),
        "final_available_ram_gb": round(ram_after.available / (1024**3), 2),
        "validation_metrics_best_epoch": diag_summary["best_epoch_metrics"],
        "scale_diagnostics": diag_summary["scale_diagnostics"],
        "per_class_validation": diag_summary["per_class_validation"],
    }

    out_file = EXP_DIR / "training_and_validation_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    print(f"\nAll artifacts recorded at {out_file}")

def run_validation_diagnostics(best_weights_path):
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

    # Load Model and run diagnostic prediction on validation images at 1280
    model = YOLO(str(best_weights_path))
    print("Running official validation on best.pt at imgsz=1280...")
    val_eval = model.val(data=str(DATA_YAML), imgsz=1280, split="val", device="cpu", verbose=True)
    val_img_paths = sorted(list(VAL_IMAGES_DIR.glob("*.jpg")))

    # Load Ground Truth (scaled to 1280x1280)
    gt_data = {}
    gt_scale_counts = {"small": 0, "medium": 0, "large": 0}
    for ip in val_img_paths:
        stem = ip.stem
        lbl_file = VAL_LABELS_DIR / f"{stem}.txt"
        boxes = []
        if lbl_file.exists():
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    # Scale to 640 equivalent for area calculation (so small < 1024 px^2 is consistent with 640 standard)
                    xyxy_640 = xywhn2xyxy(xc, yc, w, h, 640, 640)
                    area_640 = (xyxy_640[2] - xyxy_640[0]) * (xyxy_640[3] - xyxy_640[1])
                    scale = "small" if area_640 < 1024 else ("medium" if area_640 <= 9216 else "large")
                    # Actual coordinates in 1280 space
                    xyxy_1280 = xywhn2xyxy(xc, yc, w, h, 1280, 1280)
                    boxes.append({
                        "class_id": cid,
                        "class_name": CLASS_NAMES[cid],
                        "xyxy": xyxy_1280,
                        "scale": scale,
                        "matched": False
                    })
                    gt_scale_counts[scale] += 1
        gt_data[stem] = boxes

    # Predict at conf=0.25 (standard) and conf=0.05 (diagnostic)
    results_025 = model.predict(source=[str(p) for p in val_img_paths], conf=0.25, iou=0.45, device="cpu", imgsz=1280, verbose=False)
    results_005 = model.predict(source=[str(p) for p in val_img_paths], conf=0.05, iou=0.45, device="cpu", imgsz=1280, verbose=False)

    scale_stats_025 = compute_scale_recall(results_025, gt_data, conf_thresh=0.25)
    scale_stats_005 = compute_scale_recall(results_005, gt_data, conf_thresh=0.05)

    # Generate visual comparisons
    vis_dir = EXP_DIR / "error_analysis" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    generate_1280_visualizations(results_025, gt_data, val_img_paths, vis_dir)

    eval_dict = {}
    if hasattr(val_eval, "results_dict"):
        for k, v in val_eval.results_dict.items():
            try:
                eval_dict[k] = float(v)
            except Exception:
                eval_dict[k] = str(v)

    return {
        "completed_epochs": len(epochs_data),
        "best_epoch": best_epoch_idx,
        "best_epoch_metrics": best_epoch_row,
        "official_val_metrics": eval_dict,
        "scale_diagnostics": {
            "conf_025": scale_stats_025,
            "conf_005": scale_stats_005,
        },
        "per_class_validation": scale_stats_025["per_class"],
    }

def compute_scale_recall(results, raw_gt_data, conf_thresh):
    gt_detail = {}
    for stem, boxes in raw_gt_data.items():
        gt_detail[stem] = [{k: v for k, v in b.items()} for b in boxes]

    scale_stats = {
        "small": {"total": 0, "detected": 0},
        "medium": {"total": 0, "detected": 0},
        "large": {"total": 0, "detected": 0},
    }
    per_class = {c: {"gt": 0, "tp": 0, "fn": 0} for c in CLASS_NAMES}

    for stem, gts in gt_detail.items():
        for g in gts:
            sc = g["scale"]
            scale_stats[sc]["total"] += 1
            per_class[g["class_name"]]["gt"] += 1

    for res in results:
        stem = Path(res.path).stem
        gts = gt_detail[stem]
        boxes = res.boxes
        if len(boxes) == 0: continue

        for pb, pc, pconf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()):
            best_iou = 0.0
            best_idx = -1
            for idx, g in enumerate(gts):
                iou = box_iou(pb, g["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= 0.5 and pc == gts[best_idx]["class_id"]:
                if not gts[best_idx]["matched"]:
                    gts[best_idx]["matched"] = True
                    sc = gts[best_idx]["scale"]
                    scale_stats[sc]["detected"] += 1
                    per_class[CLASS_NAMES[pc]]["tp"] += 1

    for sc in scale_stats:
        tot = scale_stats[sc]["total"]
        det = scale_stats[sc]["detected"]
        scale_stats[sc]["recall"] = round(det / tot if tot > 0 else 0.0, 4)

    for c in per_class:
        gt = per_class[c]["gt"]
        tp = per_class[c]["tp"]
        per_class[c]["fn"] = gt - tp
        per_class[c]["recall"] = round(tp / gt if gt > 0 else 0.0, 4)

    return {
        "scales": scale_stats,
        "per_class": per_class
    }

def generate_1280_visualizations(results_025, gt_data, val_img_paths, vis_dir):
    selected_indices = [0, 4, 8, 12, 18, 25]
    for idx in selected_indices:
        if idx >= len(val_img_paths): continue
        ip = val_img_paths[idx]
        stem = ip.stem
        res = results_025[idx]

        img = cv2.imread(str(ip))
        if img is None: continue
        # Resize to 1280x1280 for visual alignment
        img_1280 = cv2.resize(img, (1280, 1280))
        h, w = 1280, 1280

        canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
        left = img_1280.copy()
        right = img_1280.copy()

        # Draw Ground Truth on Left
        for g in gt_data[stem]:
            x1, y1, x2, y2 = map(int, g["xyxy"])
            cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(left, f"GT: {g['class_name']}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw Predictions on Right
        pred_boxes = res.boxes
        if len(pred_boxes) > 0:
            for pb, pc, pconf in zip(pred_boxes.xyxy.cpu().numpy(),
                                     pred_boxes.cls.cpu().numpy().astype(int),
                                     pred_boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, pb)
                pname = CLASS_NAMES[pc]
                best_iou = 0.0
                match_class = -1
                for g in gt_data[stem]:
                    iou = box_iou(pb, g["xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        match_class = g["class_id"]

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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(left, f"1280 Ground Truth ({stem})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(right, f"1280 Predictions ({stem})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        canvas[:, :w] = left
        canvas[:, w:] = right

        out_vis = vis_dir / f"val_1280_comparison_{stem}.jpg"
        cv2.imwrite(str(out_vis), canvas)

if __name__ == "__main__":
    train_1280()
