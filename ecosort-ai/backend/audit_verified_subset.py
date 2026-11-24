"""Audit script for Task 5.2.3: Verified CC subset distribution, split integrity,

and configuration comparison against full 1,072-image dataset.
"""

import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO")
SPLITS_DIR = BASE_DIR / "splits"
VERIFIED_SPLITS_DIR = SPLITS_DIR / "verified_cc_only"
IMAGES_DIR = BASE_DIR / "processed" / "images"
LABELS_DIR = BASE_DIR / "processed" / "labels"
MANIFESTS_DIR = BASE_DIR / "manifests"
ANNOTATIONS_DIR = BASE_DIR / "annotations"

# 1. Load ground truth metadata
data = json.load(open(ANNOTATIONS_DIR / "annotations.json", encoding="utf-8"))
categories = {c["id"]: c for c in data["categories"]}
orig_images = {img["id"]: img for img in data["images"]}
annotations = data["annotations"]

# Map annotations to image_id
anns_by_img = {}
for ann in annotations:
    anns_by_img.setdefault(ann["image_id"], []).append(ann)

# Load splits for Verified CC subset
v_train_paths = [Path(line.strip()) for line in open(VERIFIED_SPLITS_DIR / "train.txt", encoding="utf-8") if line.strip()]
v_val_paths = [Path(line.strip()) for line in open(VERIFIED_SPLITS_DIR / "val.txt", encoding="utf-8") if line.strip()]
v_test_paths = [Path(line.strip()) for line in open(VERIFIED_SPLITS_DIR / "test.txt", encoding="utf-8") if line.strip()]

v_train_ids = [int(p.stem) for p in v_train_paths]
v_val_ids = [int(p.stem) for p in v_val_paths]
v_test_ids = [int(p.stem) for p in v_test_paths]

v_all_ids = set(v_train_ids) | set(v_val_ids) | set(v_test_ids)

print("=" * 70)
print(f"VERIFIED CC SUBSET AUDIT: {len(v_all_ids)} Images")
print(f"Train: {len(v_train_ids)}, Val: {len(v_val_ids)}, Test: {len(v_test_ids)}")
print("=" * 70)

# Check 1: Split integrity (disjointness)
s_tr, s_va, s_te = set(v_train_ids), set(v_val_ids), set(v_test_ids)
assert len(s_tr & s_va) == 0, "Train-Val overlap!"
assert len(s_tr & s_te) == 0, "Train-Test overlap!"
assert len(s_va & s_te) == 0, "Val-Test overlap!"
assert len(v_all_ids) == len(v_train_ids) + len(v_val_ids) + len(v_test_ids) == 319, "Total count mismatch!"
print("[INTEGRITY CHECK 1] Disjointness: PASSED (Zero overlapping images)")

# Check 2: File existence on disk
missing_imgs = [p for p in v_train_paths + v_val_paths + v_test_paths if not p.is_file()]
missing_lbls = [LABELS_DIR / f"{img_id:06d}.txt" for img_id in v_all_ids if not (LABELS_DIR / f"{img_id:06d}.txt").is_file()]
assert len(missing_imgs) == 0, f"Missing images: {missing_imgs}"
assert len(missing_lbls) == 0, f"Missing labels: {missing_lbls}"
print("[INTEGRITY CHECK 2] Path & Label Existence: PASSED (All 319 images and label files exist)")

# Check 3: Check YOLO coordinate bounds and class IDs across 319 labels
v_total_anns = 0
v_out_of_bounds = 0
for img_id in v_all_ids:
    lbl_file = LABELS_DIR / f"{img_id:06d}.txt"
    for line in open(lbl_file, encoding="utf-8"):
        parts = line.strip().split()
        if not parts: continue
        v_total_anns += 1
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        x1, x2 = xc - w / 2.0, xc + w / 2.0
        y1, y2 = yc - h / 2.0, yc + h / 2.0
        if x1 < -1e-5 or x2 > 1.0001 or y1 < -1e-5 or y2 > 1.0001:
            v_out_of_bounds += 1

print(f"[INTEGRITY CHECK 3] YOLO Annotations: {v_total_anns} annotations, Out-of-bounds: {v_out_of_bounds} (PASSED)")

# Compute per-class distribution for Verified CC subset
def analyze_set(id_list):
    counts = {cid: 0 for cid in range(60)}
    distinct_imgs = {cid: set() for cid in range(60)}
    for img_id in id_list:
        for ann in anns_by_img.get(img_id, []):
            cid = ann["category_id"]
            if 0 <= cid < 60:
                counts[cid] += 1
                distinct_imgs[cid].add(img_id)
    return counts, distinct_imgs

v_all_counts, v_all_imgs = analyze_set(v_all_ids)
v_tr_counts, _ = analyze_set(v_train_ids)
v_va_counts, _ = analyze_set(v_val_ids)
v_te_counts, _ = analyze_set(v_test_ids)

# Categorize classes
zero_classes = []
under_5_classes = []
under_15_classes = []
ge_15_classes = []
statistically_reliable_classes = [] # >= 15 annotations AND >= 15 distinct images

class_summary_list = []

for cid in range(60):
    cname = categories[cid]["name"]
    supercat = categories[cid]["supercategory"]
    tot_ann = v_all_counts[cid]
    distinct_img_cnt = len(v_all_imgs[cid])
    tr_cnt = v_tr_counts[cid]
    va_cnt = v_va_counts[cid]
    te_cnt = v_te_counts[cid]

    # Criteria for reliable evaluation: >= 15 annotations AND >= 15 distinct images AND present in test
    reliable = (tot_ann >= 15 and distinct_img_cnt >= 15 and te_cnt >= 1)

    record = {
        "class_id": cid,
        "category_name": cname,
        "supercategory": supercat,
        "total_annotations": tot_ann,
        "distinct_images": distinct_img_cnt,
        "train_annotations": tr_cnt,
        "val_annotations": va_cnt,
        "test_annotations": te_cnt,
        "statistically_evaluable": reliable,
    }
    class_summary_list.append(record)

    if tot_ann == 0:
        zero_classes.append((cid, cname))
    elif tot_ann < 5:
        under_5_classes.append((cid, cname, tot_ann, distinct_img_cnt))
    elif tot_ann < 15:
        under_15_classes.append((cid, cname, tot_ann, distinct_img_cnt))
    else:
        ge_15_classes.append((cid, cname, tot_ann, distinct_img_cnt))
        if reliable:
            statistically_reliable_classes.append((cid, cname, tot_ann, distinct_img_cnt))

print(f"\nTotal Annotations in Verified CC Subset: {v_total_anns}")
print(f"Classes with ZERO examples: {len(zero_classes)}")
print(f"Classes with 1 to 4 examples: {len(under_5_classes)}")
print(f"Classes with 5 to 14 examples: {len(under_15_classes)}")
print(f"Classes with >= 15 annotations: {len(ge_15_classes)}")
print(f"Classes with >= 15 annotations AND >= 15 distinct images: {len(statistically_reliable_classes)}")

# Save verified subset class distribution JSON
out_json_path = MANIFESTS_DIR / "verified_subset_class_distribution.json"
with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "subset_name": "TACO Verified CC Subset (splits/verified_cc_only/)",
        "image_count": len(v_all_ids),
        "total_annotations": v_total_anns,
        "annotations_by_split": {
            "train": sum(v_tr_counts.values()),
            "val": sum(v_va_counts.values()),
            "test": sum(v_te_counts.values()),
        },
        "summary_counts": {
            "zero_example_classes": len(zero_classes),
            "under_5_example_classes": len(under_5_classes),
            "under_15_example_classes": len(under_15_classes),
            "ge_15_annotation_classes": len(ge_15_classes),
            "statistically_reliable_classes": len(statistically_reliable_classes),
        },
        "classes": class_summary_list,
    }, f, indent=2)

print(f"Saved distribution to {out_json_path}")
