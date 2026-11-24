import cv2
from pathlib import Path
from ultralytics import YOLO
import json

VAL_IMAGES_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\images\val")
VAL_LABELS_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\labels\val")
BEST_WEIGHTS = Path(r"D:\EcoSort-Resources\experiments\yolo11n_1280_baseline\weights\best.pt")
EXP_DIR = Path(r"D:\EcoSort-Resources\experiments\yolo11n_1280_baseline")

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

def evaluate():
    model = YOLO(str(BEST_WEIGHTS))
    val_img_paths = sorted(list(VAL_IMAGES_DIR.glob("*.jpg")))
    
    scale_results = {}
    for conf_thresh in [0.25, 0.05]:
        results = model.predict(source=[str(p) for p in val_img_paths], conf=conf_thresh, iou=0.45, device="cpu", imgsz=1280, verbose=False)
        
        scale_stats = {
            "small": {"total": 0, "detected": 0},
            "medium": {"total": 0, "detected": 0},
            "large": {"total": 0, "detected": 0}
        }
        per_class = {c: {"gt": 0, "tp": 0, "fn": 0} for c in CLASS_NAMES}

        for res in results:
            img_path = Path(res.path)
            stem = img_path.stem
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
                        # In 640 equivalent space:
                        area_640 = (w * 640) * (h * 640)
                        scale = "small" if area_640 < 1024 else ("medium" if area_640 <= 9216 else "large")
                        # In original pixel space:
                        x1 = (xc - w / 2) * orig_w
                        y1 = (yc - h / 2) * orig_h
                        x2 = (xc + w / 2) * orig_w
                        y2 = (yc + h / 2) * orig_h
                        gts.append({"cid": cid, "cname": CLASS_NAMES[cid], "box": [x1, y1, x2, y2], "scale": scale, "matched": False})
                        scale_stats[scale]["total"] += 1
                        per_class[CLASS_NAMES[cid]]["gt"] += 1

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

        conf_key = f"conf_{str(conf_thresh).replace('.', '')}"
        scale_results[conf_key] = {
            "scales": scale_stats,
            "per_class": per_class
        }

    # Also update visualizations
    vis_dir = EXP_DIR / "error_analysis" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    res_025 = model.predict(source=[str(p) for p in val_img_paths], conf=0.25, iou=0.45, device="cpu", imgsz=1280, verbose=False)
    
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

        # Load GT
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
        cv2.putText(right, f"1280 Pred ({stem})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        h_disp = 640
        w_disp = int(orig_w * (640 / orig_h))
        left_r = cv2.resize(left, (w_disp, h_disp))
        right_r = cv2.resize(right, (w_disp, h_disp))
        canvas = np.hstack([left_r, right_r])

        out_vis = vis_dir / f"val_1280_comparison_{stem}.jpg"
        cv2.imwrite(str(out_vis), canvas)

    # Read existing summary and update it
    summary_path = EXP_DIR / "training_and_validation_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    summary["scale_diagnostics"] = scale_results
    summary["per_class_validation"] = scale_results["conf_025"]["per_class"]

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("SUCCESS: Scale results computed and saved to summary JSON.")
    print("Conf 0.25 Scales:", json.dumps(scale_results["conf_025"]["scales"], indent=2))
    print("Conf 0.05 Scales:", json.dumps(scale_results["conf_005"]["scales"], indent=2))

if __name__ == "__main__":
    import numpy as np
    evaluate()
