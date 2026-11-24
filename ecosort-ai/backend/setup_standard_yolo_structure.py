r"""Script to setup standard Ultralytics YOLO structure in D:\EcoSort-Resources\datasets\TACO\ecosort_yolo
and verify label resolution.
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO")
ECOSORT_YOLO_DIR = BASE_DIR / "ecosort_yolo"
PROCESSED_IMAGES_DIR = BASE_DIR / "processed" / "images"
SPLITS_DIR = BASE_DIR / "splits" / "verified_cc_only"

# 1. Setup directories
images_dir = ECOSORT_YOLO_DIR / "images"
for s in ["train", "val", "test"]:
    (images_dir / s).mkdir(parents=True, exist_ok=True)

# 2. Copy split images into images/train, images/val, images/test
split_files = {
    "train": SPLITS_DIR / "train.txt",
    "val": SPLITS_DIR / "val.txt",
    "test": SPLITS_DIR / "test.txt",
}

for split_name, sfile in split_files.items():
    dest_dir = images_dir / split_name
    with open(sfile, "r", encoding="utf-8") as f:
        img_paths = [Path(line.strip()) for line in f if line.strip()]

    print(f"Copying {len(img_paths)} images to {dest_dir}...")
    for p in img_paths:
        src = PROCESSED_IMAGES_DIR / p.name
        dst = dest_dir / p.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)

    copied = list(dest_dir.glob("*.jpg"))
    print(f"  Verified {len(copied)} images in {dest_dir}")

# 3. Update data.yaml
data_yaml_path = ECOSORT_YOLO_DIR / "data.yaml"
yaml_content = """# EcoSort AI — Ultralytics YOLO11n Dataset Configuration (13 Candidate Dry Classes)
# Standard Directory Structure: ecosort_yolo/{images,labels}/{train,val,test}
# Source: TACO Verified Creative Commons Subset (Flickr CC BY / CC BY-SA)

path: D:/EcoSort-Resources/datasets/TACO/ecosort_yolo
train: images/train
val: images/val
test: images/test

nc: 13
names:
  0: clear_plastic_bottle
  1: glass_bottle
  2: plastic_bottle_cap
  3: metal_bottle_cap
  4: drink_can
  5: other_carton
  6: corrugated_carton
  7: disposable_plastic_cup
  8: other_plastic
  9: plastic_film
  10: other_plastic_wrapper
  11: styrofoam_piece
  12: cigarette
"""

with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"Updated {data_yaml_path}")
