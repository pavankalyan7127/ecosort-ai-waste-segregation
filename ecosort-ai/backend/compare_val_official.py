from pathlib import Path
from ultralytics import YOLO
import json

DATA_YAML = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\data.yaml")
WEIGHTS_640 = Path(r"D:\EcoSort-Resources\experiments\yolo11n_baseline_13class\weights\best.pt")
WEIGHTS_1280 = Path(r"D:\EcoSort-Resources\experiments\yolo11n_1280_baseline\weights\best.pt")

print("--- Validating 640 Best Model at imgsz=640 ---")
m640 = YOLO(str(WEIGHTS_640))
val640 = m640.val(data=str(DATA_YAML), imgsz=640, split="val", device="cpu", verbose=True)

print("\n--- Validating 1280 Best Model at imgsz=1280 ---")
m1280 = YOLO(str(WEIGHTS_1280))
val1280 = m1280.val(data=str(DATA_YAML), imgsz=1280, split="val", device="cpu", verbose=True)

names = m640.names

summary = {
    "model_640": {
        "all": {
            "p": float(val640.results_dict["metrics/precision(B)"]),
            "r": float(val640.results_dict["metrics/recall(B)"]),
            "map50": float(val640.results_dict["metrics/mAP50(B)"]),
            "map50_95": float(val640.results_dict["metrics/mAP50-95(B)"]),
        },
        "per_class": {
            names[i]: {
                "map50": float(val640.box.maps[i]) if hasattr(val640.box, "maps") else None,
                "map50_95": float(val640.box.maps[i]) if hasattr(val640.box, "maps") else None,
            }
            for i in range(len(names))
        }
    },
    "model_1280": {
        "all": {
            "p": float(val1280.results_dict["metrics/precision(B)"]),
            "r": float(val1280.results_dict["metrics/recall(B)"]),
            "map50": float(val1280.results_dict["metrics/mAP50(B)"]),
            "map50_95": float(val1280.results_dict["metrics/mAP50-95(B)"]),
        },
        "per_class": {
            names[i]: {
                "map50": float(val1280.box.maps[i]) if hasattr(val1280.box, "maps") else None,
                "map50_95": float(val1280.box.maps[i]) if hasattr(val1280.box, "maps") else None,
            }
            for i in range(len(names))
        }
    }
}

out_p = Path(r"D:\EcoSort-Resources\experiments\yolo11n_1280_baseline\resolution_comparison_official.json")
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved comparison to", out_p)
