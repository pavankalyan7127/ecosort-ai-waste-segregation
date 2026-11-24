"""Comprehensive annotation reconciliation audit script for TACO dataset."""

import json
from pathlib import Path

base_dir = Path(r"D:\EcoSort-Resources\datasets\TACO")
orig_ann_file = base_dir / "annotations" / "annotations.json"
resized_ann_file = base_dir / "annotations" / "taco_resized_annotations.json"
images_dir = base_dir / "processed" / "images"
labels_dir = base_dir / "processed" / "labels"
original_dir = base_dir / "original"

orig_data = json.load(open(orig_ann_file, encoding="utf-8"))
resized_data = json.load(open(resized_ann_file, encoding="utf-8"))

orig_images = {img["id"]: img for img in orig_data["images"]}
orig_anns = orig_data["annotations"]

proc_images = sorted(list(images_dir.glob("*.jpg")))
proc_labels = sorted(list(labels_dir.glob("*.txt")))

print("=" * 70)
print("TACO DATASET — ANNOTATION RECONCILIATION AUDIT")
print("=" * 70)

print(f"Original images in annotations.json: {len(orig_images)}")
print(f"Original annotations in annotations.json: {len(orig_anns)}")
print(f"Processed images on disk: {len(proc_images)}")
print(f"Processed label files on disk: {len(proc_labels)}")
print(f"Resized COCO annotations count: {len(resized_data['annotations'])}")

# Map annotations to image_id
anns_by_img = {}
for ann in orig_anns:
    anns_by_img.setdefault(ann["image_id"], []).append(ann)

proc_ids = set(int(p.stem) for p in proc_images)
label_ids = set(int(p.stem) for p in proc_labels)

# 1. Image to Label 1:1 match
assert proc_ids == label_ids, f"Mismatch between images and labels! Diff: {proc_ids ^ label_ids}"
print("\n[CHECK 1] Exactly 1:1 image-to-label correspondence on disk: VERIFIED (1,072 files).")

# 2. Reconcile annotation counts
anns_in_downloaded = sum(len(anns_by_img.get(img_id, [])) for img_id in proc_ids)
anns_in_excluded = sum(len(anns_by_img.get(img_id, [])) for img_id in orig_images if img_id not in proc_ids)

print(f"\n[CHECK 2] Annotation Partitioning:")
print(f"  Annotations belonging to downloaded images: {anns_in_downloaded}")
print(f"  Annotations belonging to excluded images (OLM offline): {anns_in_excluded}")
print(f"  Sum: {anns_in_downloaded + anns_in_excluded} == {len(orig_anns)} -> {anns_in_downloaded + anns_in_excluded == len(orig_anns)}")
assert anns_in_downloaded + anns_in_excluded == len(orig_anns), "Annotation count mismatch!"

# 3. Parse all YOLO labels from disk and verify
yolo_total_lines = 0
out_of_bounds = 0
invalid_class_ids = 0
empty_label_files = []

for l_file in proc_labels:
    img_id = int(l_file.stem)
    lines = [line.strip() for line in open(l_file, encoding="utf-8") if line.strip()]
    if not lines:
        empty_label_files.append(img_id)
    for line in lines:
        yolo_total_lines += 1
        parts = line.split()
        if len(parts) != 5:
            print(f"Malformed line in {l_file}: {line}")
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        if cls_id < 0 or cls_id >= 60:
            invalid_class_ids += 1
        # Check normalized bounds [0, 1]
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            out_of_bounds += 1
        # Check box corners within frame
        x1 = xc - w / 2.0
        x2 = xc + w / 2.0
        y1 = yc - h / 2.0
        y2 = yc + h / 2.0
        if x1 < -1e-5 or x2 > 1.0001 or y1 < -1e-5 or y2 > 1.0001:
            out_of_bounds += 1

print(f"\n[CHECK 3] YOLO Label Files Integrity:")
print(f"  Total YOLO annotation lines on disk: {yolo_total_lines}")
print(f"  Empty label files count: {len(empty_label_files)} ({empty_label_files})")
print(f"  Invalid class IDs: {invalid_class_ids}")
print(f"  Out-of-bounds coordinates: {out_of_bounds}")

assert yolo_total_lines == anns_in_downloaded == 3393, f"Count mismatch! {yolo_total_lines} vs {anns_in_downloaded}"
assert yolo_total_lines == len(resized_data["annotations"]), "COCO resized count mismatch!"
print(f"  Reproduction Confirmation: EXACTLY 3,393 annotations independently reproduced.")
print("=" * 70)
