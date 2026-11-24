r"""Inspect predictions.json to check test detections count and distribution.
"""

import json
from pathlib import Path
from collections import defaultdict

pred_path = Path(r"D:\EcoSort-Resources\experiments\yolo11n_baseline_13class\test_evaluation\predictions.json")
with open(pred_path, "r", encoding="utf-8") as f:
    preds = json.load(f)

print(f"Total predictions: {len(preds)}")
cls_counts = defaultdict(int)
conf_scores = defaultdict(list)
for p in preds:
    cat = p.get("category_id")
    score = p.get("score", 0.0)
    cls_counts[cat] += 1
    conf_scores[cat].append(score)

print("\nPredictions by category:")
for cat in sorted(cls_counts.keys()):
    scores = conf_scores[cat]
    print(f"  Category {cat}: count={cls_counts[cat]}, avg_score={sum(scores)/len(scores):.3f}, max_score={max(scores):.3f}")
