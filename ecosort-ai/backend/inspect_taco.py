"""Inspection and quantitative analysis script for official TACO annotations."""

import json
from collections import Counter, defaultdict
from pathlib import Path

annotations_path = Path(r"D:\EcoSort-Resources\datasets\TACO\annotations.json")

with open(annotations_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("TACO DATASET — OFFICIAL ANNOTATION METADATA INSPECTION")
print("=" * 70)

# Top level keys
keys = list(data.keys())
print("Top-level keys:", keys)

# Info & Licenses
info = data.get("info", {})
print("\n[INFO SECTION]:")
for k, v in info.items():
    print(f"  {k}: {v}")

licenses = data.get("licenses", [])
print("\n[LICENSES]:")
for lic in licenses:
    print(f"  id: {lic.get('id')}, name: {lic.get('name')}, url: {lic.get('url')}")

# Images
images = data.get("images", [])
total_images = len(images)
print(f"\n[IMAGES]: Total images listed in annotations = {total_images}")
if images:
    sample_img = images[0]
    print(f"  Sample image entry: id={sample_img.get('id')}, file_name={sample_img.get('file_name')}, resolution={sample_img.get('width')}x{sample_img.get('height')}")

# Categories
categories = data.get("categories", [])
total_categories = len(categories)
print(f"\n[CATEGORIES]: Total defined categories = {total_categories}")

cat_map = {c["id"]: c for c in categories}
supercategories = set(c.get("supercategory") for c in categories)
print(f"Total unique supercategories: {len(supercategories)}")
print("Supercategories:", sorted(list(supercategories)))

# Annotations
annotations = data.get("annotations", [])
total_annotations = len(annotations)
print(f"\n[ANNOTATIONS]: Total annotated objects (bounding boxes) = {total_annotations}")

if annotations:
    sample_ann = annotations[0]
    print(f"  Sample annotation entry: id={sample_ann.get('id')}, image_id={sample_ann.get('image_id')}, category_id={sample_ann.get('category_id')}, bbox={sample_ann.get('bbox')}, area={sample_ann.get('area')}")

# Image-level annotation distribution
image_ann_counts = Counter(ann["image_id"] for ann in annotations)
images_with_anns = len(image_ann_counts)
images_with_zero_anns = total_images - images_with_anns
single_obj_images = sum(1 for count in image_ann_counts.values() if count == 1)
multi_obj_images = sum(1 for count in image_ann_counts.values() if count > 1)
max_objects_in_single_image = max(image_ann_counts.values()) if image_ann_counts else 0

print(f"\n[IMAGE-LEVEL DISTRIBUTION]:")
print(f"  Images with >= 1 annotation: {images_with_anns}")
print(f"  Images with 0 annotations: {images_with_zero_anns}")
print(f"  Images with exactly 1 object: {single_obj_images} ({single_obj_images/total_images*100:.1f}%)")
print(f"  Images with multiple objects (>1): {multi_obj_images} ({multi_obj_images/total_images*100:.1f}%)")
print(f"  Maximum objects in a single image: {max_objects_in_single_image}")

# Category-level object counts
cat_counts = Counter(ann["category_id"] for ann in annotations)

# Group categories by count
sorted_cats = sorted(categories, key=lambda c: cat_counts.get(c["id"], 0), reverse=True)

print(f"\n[CATEGORY DISTRIBUTION (Ranked by Count)]:")
print(f"{'ID':<4} | {'Category Name':<32} | {'Supercategory':<22} | {'Count':<6} | {'% Total':<7}")
print("-" * 78)

zero_count_cats = []
under_10_cats = []
under_25_cats = []

for c in sorted_cats:
    cid = c["id"]
    name = c["name"]
    supercat = c.get("supercategory", "None")
    cnt = cat_counts.get(cid, 0)
    pct = (cnt / total_annotations * 100) if total_annotations else 0
    print(f"{cid:<4} | {name:<32} | {supercat:<22} | {cnt:<6} | {pct:>5.2f}%")

    if cnt == 0:
        zero_count_cats.append((cid, name))
    elif cnt < 10:
        under_10_cats.append((cid, name, cnt))
    elif cnt < 25:
        under_25_cats.append((cid, name, cnt))

print("\n" + "=" * 70)
print(f"Categories with 0 annotations ({len(zero_count_cats)}): {zero_count_cats}")
print(f"Categories with 1-9 annotations ({len(under_10_cats)}): {under_10_cats}")
print(f"Categories with 10-24 annotations ({len(under_25_cats)}): {under_25_cats}")
print("=" * 70)
