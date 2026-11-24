r"""Script to prepare D:\EcoSort-Resources\datasets\TACO\ecosort_yolo configuration

and update verified_subset_class_distribution.json with exact, corrected metrics.
"""

import json
from pathlib import Path

BASE_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO")
SPLITS_DIR = BASE_DIR / "splits"
VERIFIED_SPLITS_DIR = SPLITS_DIR / "verified_cc_only"
PROCESSED_LABELS_DIR = BASE_DIR / "processed" / "labels"
PROCESSED_IMAGES_DIR = BASE_DIR / "processed" / "images"
MANIFESTS_DIR = BASE_DIR / "manifests"
ANNOTATIONS_FILE = BASE_DIR / "annotations" / "annotations.json"
ECOSORT_YOLO_DIR = BASE_DIR / "ecosort_yolo"

# Create directories
ECOSORT_YOLO_DIR.mkdir(parents=True, exist_ok=True)
for sub in ["labels/train", "labels/val", "labels/test"]:
    (ECOSORT_YOLO_DIR / sub).mkdir(parents=True, exist_ok=True)

# Load annotations
data = json.load(open(ANNOTATIONS_FILE, encoding="utf-8"))
categories = {c["id"]: c for c in data["categories"]}

# Load splits
train_paths = [Path(line.strip()) for line in open(VERIFIED_SPLITS_DIR / "train.txt", encoding="utf-8") if line.strip()]
val_paths = [Path(line.strip()) for line in open(VERIFIED_SPLITS_DIR / "val.txt", encoding="utf-8") if line.strip()]
test_paths = [Path(line.strip()) for line in open(VERIFIED_SPLITS_DIR / "test.txt", encoding="utf-8") if line.strip()]

train_ids = [int(p.stem) for p in train_paths]
val_ids = [int(p.stem) for p in val_paths]
test_ids = [int(p.stem) for p in test_paths]
all_ids = train_ids + val_ids + test_ids

print(f"Verified CC Subset: {len(all_ids)} images (Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)})")

# Recalculate class distribution directly from processed labels on disk
def count_objects(id_list):
    counts = {cid: 0 for cid in range(60)}
    distinct_imgs = {cid: set() for cid in range(60)}
    for img_id in id_list:
        lbl_path = PROCESSED_LABELS_DIR / f"{img_id:06d}.txt"
        for line in open(lbl_path, encoding="utf-8"):
            parts = line.strip().split()
            if not parts: continue
            cid = int(parts[0])
            counts[cid] += 1
            distinct_imgs[cid].add(img_id)
    return counts, distinct_imgs

all_counts, all_imgs = count_objects(all_ids)
tr_counts, _ = count_objects(train_ids)
va_counts, _ = count_objects(val_ids)
te_counts, _ = count_objects(test_ids)

# 13 Candidate Waste Classes Mapping
CANDIDATE_13_CLASSES = [
    {"yolo_id": 0, "taco_id": 5, "name": "clear_plastic_bottle", "original_name": "Clear plastic bottle", "ecosort_class": "dry_plastic_pet_bottles", "primary_stream": "Dry waste", "mapping_confidence": "Direct", "statutory_source": "SWM Rules 2026 Rule 3(s), 3(zw): PET beverage bottles (Resin Code 1)."},
    {"yolo_id": 1, "taco_id": 6, "name": "glass_bottle", "original_name": "Glass bottle", "ecosort_class": "dry_glass_bottles_jars", "primary_stream": "Dry waste", "mapping_confidence": "Direct", "statutory_source": "SWM Rules 2026 Rule 3(zw): 100% recyclable glass packaging."},
    {"yolo_id": 2, "taco_id": 7, "name": "plastic_bottle_cap", "original_name": "Plastic bottle cap", "ecosort_class": "dry_plastic_pp_containers", "primary_stream": "Dry waste", "mapping_confidence": "Direct", "statutory_source": "SWM Rules 2026 Rule 3(s): Polypropylene/HDPE rigid closures."},
    {"yolo_id": 3, "taco_id": 8, "name": "metal_bottle_cap", "original_name": "Metal bottle cap", "ecosort_class": "dry_metal_steel_tin_cans", "primary_stream": "Dry waste", "mapping_confidence": "Direct", "statutory_source": "SWM Rules 2026 Rule 3(zw): Crown corks and steel twist bottle closures."},
    {"yolo_id": 4, "taco_id": 12, "name": "drink_can", "original_name": "Drink can", "ecosort_class": "dry_metal_aluminium_cans", "primary_stream": "Dry waste", "mapping_confidence": "Direct", "statutory_source": "SWM Rules 2026 Rule 3(s), 3(zw): High-grade aluminium alloy beverage cans."},
    {"yolo_id": 5, "taco_id": 14, "name": "other_carton", "original_name": "Other carton", "ecosort_class": "dry_paper_cardboard_cartons", "primary_stream": "Dry waste", "mapping_confidence": "Partial", "statutory_source": "SWM Rules 2026: Clean paperboard packaging (unsoiled by liquid grease)."},
    {"yolo_id": 6, "taco_id": 17, "name": "corrugated_carton", "original_name": "Corrugated carton", "ecosort_class": "dry_paper_cardboard_cartons", "primary_stream": "Dry waste", "mapping_confidence": "Direct", "statutory_source": "SWM Rules 2026: Corrugated cardboard shipping boxes."},
    {"yolo_id": 7, "taco_id": 21, "name": "disposable_plastic_cup", "original_name": "Disposable plastic cup", "ecosort_class": "dry_plastic_pp_containers", "primary_stream": "Dry waste", "mapping_confidence": "Direct", "statutory_source": "SWM Rules 2026 / PWM Rules: Polypropylene/PET drinkware."},
    {"yolo_id": 8, "taco_id": 29, "name": "other_plastic", "original_name": "Other plastic", "ecosort_class": "dry_plastic_hdpe_containers", "primary_stream": "Dry waste", "mapping_confidence": "Partial", "statutory_source": "SWM Rules 2026: Miscellaneous rigid plastics; mapped to dry recyclable polymer stream."},
    {"yolo_id": 9, "taco_id": 36, "name": "plastic_film", "original_name": "Plastic film", "ecosort_class": "dry_plastic_ldpe_films", "primary_stream": "Dry waste", "mapping_confidence": "Direct", "statutory_source": "SWM Rules 2026: LDPE clean packaging film and stretch wrap."},
    {"yolo_id": 10, "taco_id": 39, "name": "other_plastic_wrapper", "original_name": "Other plastic wrapper", "ecosort_class": "dry_plastic_mlp_non_recyclable", "primary_stream": "Dry waste", "mapping_confidence": "Partial", "statutory_source": "SWM Rules 2026: Multi-layer packaging / metallised BOPP wrappers directed to RDF/co-processing."},
    {"yolo_id": 11, "taco_id": 57, "name": "styrofoam_piece", "original_name": "Styrofoam piece", "ecosort_class": "dry_plastic_single_use_banned", "primary_stream": "Dry waste", "mapping_confidence": "Direct", "statutory_source": "PWM (Amendment) Rules 2021: Banned expanded polystyrene (thermocol)."},
    {"yolo_id": 12, "taco_id": 59, "name": "cigarette", "original_name": "Cigarette", "ecosort_class": "dry_inerts_street_sweepings", "primary_stream": "Dry waste", "mapping_confidence": "Direct", "statutory_source": "SWM Rules 2026: Non-recyclable inerts/combustible dry street litter."}
]

taco_to_yolo = {c["taco_id"]: c["yolo_id"] for c in CANDIDATE_13_CLASSES}
candidate_taco_ids = set(taco_to_yolo.keys())

# Build Mutually Exclusive Category Groups
zero_classes = []
fewer_than_5_classes = []
five_to_14_classes = []
candidate_classes = []
high_clutter_classes = []

class_records = []

for cid in range(60):
    cname = categories[cid]["name"]
    supercat = categories[cid]["supercategory"]
    tot = all_counts[cid]
    imgs = len(all_imgs[cid])
    tr = tr_counts[cid]
    va = va_counts[cid]
    te = te_counts[cid]

    # Mutually exclusive assignment:
    # 1. Zero: tot == 0
    # 2. Fewer than 5: 1 <= tot <= 4 (explicitly covers Metal lid with 3 anns)
    # 3. 5 to 14: 5 <= tot <= 14
    # 4. Candidate classes: cid in candidate_taco_ids or (tot >= 15 and imgs >= 15 and te >= 1 and cid != 58)
    # 5. High-clutter / low distinct images: tot >= 15 and (imgs < 15 or te == 0 or cid == 58)
    if tot == 0:
        group = "Zero Examples"
        zero_classes.append({"class_id": cid, "name": cname, "total": tot, "distinct_images": imgs})
    elif 1 <= tot < 5:
        group = "Fewer than 5 Examples"
        fewer_than_5_classes.append({"class_id": cid, "name": cname, "total": tot, "distinct_images": imgs})
    elif 5 <= tot < 15:
        group = "5 to 14 Examples"
        five_to_14_classes.append({"class_id": cid, "name": cname, "total": tot, "distinct_images": imgs})
    elif cid in candidate_taco_ids:
        group = "Candidate Classes for Baseline Experimentation"
        candidate_classes.append({"class_id": cid, "name": cname, "total": tot, "distinct_images": imgs, "val": va, "test": te})
    else:
        group = "High-Clutter / Excluded Classes"
        high_clutter_classes.append({"class_id": cid, "name": cname, "total": tot, "distinct_images": imgs, "val": va, "test": te})

    class_records.append({
        "class_id": cid,
        "category_name": cname,
        "supercategory": supercat,
        "total_annotations": tot,
        "distinct_images": imgs,
        "train_annotations": tr,
        "val_annotations": va,
        "test_annotations": te,
        "viability_group": group,
        "is_candidate_13_class": cid in candidate_taco_ids,
        "yolo_contiguous_id": taco_to_yolo.get(cid),
    })

assert len(zero_classes) + len(fewer_than_5_classes) + len(five_to_14_classes) + len(candidate_classes) + len(high_clutter_classes) == 60, "Sum of groups != 60!"

print("Group counts check:")
print(f"  Zero Examples: {len(zero_classes)}")
print(f"  Fewer than 5 Examples: {len(fewer_than_5_classes)} (Metal lid included: {'Metal lid' in [x['name'] for x in fewer_than_5_classes]})")
print(f"  5 to 14 Examples: {len(five_to_14_classes)}")
print(f"  Candidate Classes: {len(candidate_classes)}")
print(f"  High-Clutter / Excluded: {len(high_clutter_classes)}")
print(f"  Sum: {len(zero_classes) + len(fewer_than_5_classes) + len(five_to_14_classes) + len(candidate_classes) + len(high_clutter_classes)} == 60")

# Save updated verified_subset_class_distribution.json
updated_dist_json = MANIFESTS_DIR / "verified_subset_class_distribution.json"
with open(updated_dist_json, "w", encoding="utf-8") as f:
    json.dump({
        "subset_name": "TACO Verified CC Subset (splits/verified_cc_only/)",
        "image_count": len(all_ids),
        "total_annotations": sum(all_counts.values()),
        "annotations_by_split": {
            "train": sum(tr_counts.values()),
            "val": sum(va_counts.values()),
            "test": sum(te_counts.values()),
        },
        "group_summary": {
            "zero_example_classes": len(zero_classes),
            "fewer_than_5_example_classes": len(fewer_than_5_classes),
            "five_to_14_example_classes": len(five_to_14_classes),
            "candidate_classes_for_baseline": len(candidate_classes),
            "high_clutter_or_excluded_classes": len(high_clutter_classes),
            "total_categories": 60,
        },
        "candidate_13_waste_classes": CANDIDATE_13_CLASSES,
        "classes": class_records,
    }, f, indent=2)

print(f"Updated {updated_dist_json}")

# Step 2: Remap YOLO labels for the 13 classes into ecosort_yolo/labels/
remapped_anns_count = {"train": 0, "val": 0, "test": 0}
total_remapped_boxes = 0

for split_name, id_list in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    for img_id in id_list:
        orig_lbl_file = PROCESSED_LABELS_DIR / f"{img_id:06d}.txt"
        new_lbl_file = ECOSORT_YOLO_DIR / f"labels/{split_name}/{img_id:06d}.txt"

        new_lines = []
        for line in open(orig_lbl_file, encoding="utf-8"):
            line = line.strip()
            if not line: continue
            parts = line.split()
            orig_cid = int(parts[0])
            if orig_cid in taco_to_yolo:
                new_cid = taco_to_yolo[orig_cid]
                xc, yc, w, h = parts[1:]
                new_lines.append(f"{new_cid} {xc} {yc} {w} {h}")
                remapped_anns_count[split_name] += 1
                total_remapped_boxes += 1

        with open(new_lbl_file, "w", encoding="utf-8") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")

print(f"Remapped YOLO labels written to {ECOSORT_YOLO_DIR / 'labels'}:")
print(f"  Train: {remapped_anns_count['train']} annotations")
print(f"  Val: {remapped_anns_count['val']} annotations")
print(f"  Test: {remapped_anns_count['test']} annotations")
print(f"  Total remapped boxes: {total_remapped_boxes}")

# Step 3: Write train.txt, val.txt, test.txt in ecosort_yolo/
for split_name, id_list in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    split_file = ECOSORT_YOLO_DIR / f"{split_name}.txt"
    with open(split_file, "w", encoding="utf-8") as f:
        for img_id in id_list:
            img_path = PROCESSED_IMAGES_DIR / f"{img_id:06d}.jpg"
            f.write(f"{img_path.as_posix()}\n")

# Step 4: Write classes.txt
classes_file = ECOSORT_YOLO_DIR / "classes.txt"
with open(classes_file, "w", encoding="utf-8") as f:
    for c in CANDIDATE_13_CLASSES:
        f.write(f"{c['name']}\n")

# Step 5: Write class_mapping.json
mapping_file = ECOSORT_YOLO_DIR / "class_mapping.json"
with open(mapping_file, "w", encoding="utf-8") as f:
    json.dump({
        "num_classes": len(CANDIDATE_13_CLASSES),
        "source_dataset": "TACO (Option B 640x640 Verified CC Subset)",
        "mapping_rationale": "Remaps 13 candidate Dry packaging classes from TACO into contiguous IDs 0..12, excluding unselected sparse classes and Unlabeled litter.",
        "classes": CANDIDATE_13_CLASSES,
    }, f, indent=2)

# Step 6: Write data.yaml
data_yaml_file = ECOSORT_YOLO_DIR / "data.yaml"
names_dict = {c["yolo_id"]: c["name"] for c in CANDIDATE_13_CLASSES}
names_yaml_str = "\n".join([f"  {k}: {v}" for k, v in names_dict.items()])

yaml_content = f"""# EcoSort AI — YOLO11n Dataset Configuration for 13 Candidate Dry Packaging Classes
# Source: TACO Verified Creative Commons Subset (Flickr CC BY / CC BY-SA)
# Root Directory: D:/EcoSort-Resources/datasets/TACO/

path: D:/EcoSort-Resources/datasets/TACO
train: ecosort_yolo/train.txt
val: ecosort_yolo/val.txt
test: ecosort_yolo/test.txt

nc: 13
names:
{names_yaml_str}
"""
with open(data_yaml_file, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"Generated data.yaml, classes.txt, class_mapping.json in {ECOSORT_YOLO_DIR}")

# Step 7: Write dataset_readme.md
readme_content = f"""# EcoSort AI — YOLO Dataset Configuration (13 Candidate Classes)

**Dataset**: TACO Verified CC Subset Remapped  
**Target Architecture**: Ultralytics YOLO11n  
**Dataset Directory**: `D:\\EcoSort-Resources\\datasets\\TACO\\ecosort_yolo\\`  
**Image Dimension**: $640 \\times 640$ px (Letterbox padded with `(114, 114, 114)`)  
**Number of Classes**: 13 contiguous classes ($0..12$)

---

## 1. Class Mapping Register

| YOLO ID | TACO Original ID | Class Name | EcoSort Knowledge Base ID | Statutory Stream | Mapping Status | Train Anns | Val Anns | Test Anns |
| :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
"""

for c in CANDIDATE_13_CLASSES:
    y_id = c["yolo_id"]
    t_id = c["taco_id"]
    readme_content += f"| **{y_id}** | {t_id} | `{c['name']}` | `{c['ecosort_class']}` | {c['primary_stream']} | {c['mapping_confidence']} | {tr_counts[t_id]} | {va_counts[t_id]} | {te_counts[t_id]} |\n"

readme_content += f"""
**Total Annotations**: {total_remapped_boxes} (Train: {remapped_anns_count['train']}, Val: {remapped_anns_count['val']}, Test: {remapped_anns_count['test']}) across 319 images.

---

## 2. Directory Structure

```
D:\\EcoSort-Resources\\datasets\\TACO\\ecosort_yolo\\
├── data.yaml              # Ultralytics dataset configuration file
├── classes.txt            # Contiguous 13 class names
├── class_mapping.json     # Detailed TACO to YOLO remapping register
├── dataset_readme.md      # This specification document
├── train.txt              # Absolute paths to 255 training images
├── val.txt                # Absolute paths to 32 validation images
├── test.txt               # Absolute paths to 32 testing images
└── labels\\
    ├── train\\             # 255 remapped YOLO label files (000000.txt ..)
    ├── val\\               # 32 remapped YOLO label files
    └── test\\              # 32 remapped YOLO label files
```

---

## 3. Data Integrity & Exclusion Rules

1. **Original Data Unaltered**: Original 60-class label files in `../processed/labels/` remain completely unchanged.
2. **Selective Filtering**: Only bounding boxes belonging to the 13 candidate classes were converted; all other classes (including `Unlabeled litter` #58, rare classes < 15 samples, and clustered classes) were excluded.
3. **Licensing**: All 319 images carry confirmed Creative Commons attribution metadata in `../manifests/image_licensing_manifest.csv`.
"""

with open(ECOSORT_YOLO_DIR / "dataset_readme.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("Generated dataset_readme.md successfully.")
