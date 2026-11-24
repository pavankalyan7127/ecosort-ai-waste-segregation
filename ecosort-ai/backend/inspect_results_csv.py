r"""Inspect results.csv to find best epoch and loss trends.
"""

import csv
from pathlib import Path

csv_path = Path(r"D:\EcoSort-Resources\experiments\yolo11n_baseline_13class\results.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows in results.csv: {len(rows)}")

best_map50 = 0.0
best_epoch = None
for r in rows:
    epoch = int(r["epoch"].strip())
    map50 = float(r["metrics/mAP50(B)"].strip())
    map50_95 = float(r["metrics/mAP50-95(B)"].strip())
    p = float(r["metrics/precision(B)"].strip())
    r_val = float(r["metrics/recall(B)"].strip())
    train_loss = float(r["train/box_loss"].strip()) + float(r["train/cls_loss"].strip()) + float(r["train/dfl_loss"].strip())
    val_loss = float(r["val/box_loss"].strip()) + float(r["val/cls_loss"].strip()) + float(r["val/dfl_loss"].strip())

    if map50 > best_map50:
        best_map50 = map50
        best_epoch = (epoch, map50, map50_95, p, r_val, train_loss, val_loss)

print("Best Validation Epoch:")
print(f"  Epoch: {best_epoch[0]}")
print(f"  mAP@50: {best_epoch[1]:.4f}")
print(f"  mAP@50-95: {best_epoch[2]:.4f}")
print(f"  Precision: {best_epoch[3]:.4f}")
print(f"  Recall: {best_epoch[4]:.4f}")
print(f"  Train Loss: {best_epoch[5]:.4f}")
print(f"  Val Loss: {best_epoch[6]:.4f}")

# First 5, middle 5, last 5 epochs
print("\nProgression checkpoints (Epoch 1, 10, 25, 40, 50):")
for ep in [1, 10, 25, 40, 50]:
    r = rows[ep - 1]
    print(f"  Ep {ep:2d}: train_loss={float(r['train/box_loss'])+float(r['train/cls_loss']):.3f}, val_mAP50={float(r['metrics/mAP50(B)']):.4f}, val_mAP50-95={float(r['metrics/mAP50-95(B)']):.4f}, P={float(r['metrics/precision(B)']):.4f}, R={float(r['metrics/recall(B)']):.4f}")
