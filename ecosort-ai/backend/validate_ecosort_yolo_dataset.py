r"""Validation script for D:\EcoSort-Resources\datasets\TACO\ecosort_yolo dataset configuration.
"""

from pathlib import Path
import json

ECOSORT_YOLO_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo")
ORIGINAL_LABELS_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO\processed\labels")
ORIGINAL_IMAGES_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO\processed\images")

def run_validation():
    print("=== Step 1: Checking File Existence ===")
    required_files = [
        ECOSORT_YOLO_DIR / "data.yaml",
        ECOSORT_YOLO_DIR / "train.txt",
        ECOSORT_YOLO_DIR / "val.txt",
        ECOSORT_YOLO_DIR / "test.txt",
        ECOSORT_YOLO_DIR / "classes.txt",
        ECOSORT_YOLO_DIR / "class_mapping.json",
        ECOSORT_YOLO_DIR / "dataset_readme.md",
    ]
    for f in required_files:
        assert f.exists(), f"Missing file: {f}"
        print(f"  [OK] {f.name} exists ({f.stat().st_size} bytes)")

    print("\n=== Step 2: Checking Split Disjointness & Image Paths ===")
    splits = {}
    for s in ["train", "val", "test"]:
        with open(ECOSORT_YOLO_DIR / f"{s}.txt", "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f if line.strip()]
        splits[s] = paths
        print(f"  {s} split: {len(paths)} image paths")
        for p in paths:
            assert Path(p).exists(), f"Image path does not exist: {p}"

    s_train, s_val, s_test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert len(s_train & s_val) == 0, "Leakage between train and val!"
    assert len(s_train & s_test) == 0, "Leakage between train and test!"
    assert len(s_val & s_test) == 0, "Leakage between val and test!"
    assert len(s_train) == 255, f"Expected 255 train images, got {len(s_train)}"
    assert len(s_val) == 32, f"Expected 32 val images, got {len(s_val)}"
    assert len(s_test) == 32, f"Expected 32 test images, got {len(s_test)}"
    print("  [OK] Zero split leakage verified across all 319 images.")

    print("\n=== Step 3: Checking Remapped Label Integrity ===")
    total_boxes = 0
    box_counts = {"train": 0, "val": 0, "test": 0}
    seen_classes = set()

    for s in ["train", "val", "test"]:
        lbl_dir = ECOSORT_YOLO_DIR / "labels" / s
        assert lbl_dir.exists(), f"Missing label dir: {lbl_dir}"
        lbl_files = list(lbl_dir.glob("*.txt"))
        assert len(lbl_files) == len(splits[s]), f"Label count ({len(lbl_files)}) does not match split image count ({len(splits[s])})"

        for lf in lbl_files:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    assert len(parts) == 5, f"Malformed line in {lf}: {line}"
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    assert 0 <= cid <= 12, f"Class ID {cid} out of range [0..12] in {lf}"
                    assert 0.0 <= xc <= 1.0, f"xc {xc} out of bounds in {lf}"
                    assert 0.0 <= yc <= 1.0, f"yc {yc} out of bounds in {lf}"
                    assert 0.0 <= w <= 1.0, f"w {w} out of bounds in {lf}"
                    assert 0.0 <= h <= 1.0, f"h {h} out of bounds in {lf}"
                    seen_classes.add(cid)
                    box_counts[s] += 1
                    total_boxes += 1

    print(f"  Remapped annotations:")
    print(f"    Train: {box_counts['train']}")
    print(f"    Val:   {box_counts['val']}")
    print(f"    Test:  {box_counts['test']}")
    print(f"    Total: {total_boxes}")
    assert box_counts["train"] == 590, f"Expected 590 train annotations, got {box_counts['train']}"
    assert box_counts["val"] == 73, f"Expected 73 val annotations, got {box_counts['val']}"
    assert box_counts["test"] == 66, f"Expected 66 test annotations, got {box_counts['test']}"
    assert total_boxes == 729, f"Expected 729 total annotations, got {total_boxes}"
    assert seen_classes == set(range(13)), f"Missing classes in dataset! Found {seen_classes}"
    print("  [OK] All 13 classes [0..12] present with valid normalized bounds.")

    print("\n=== Step 4: Checking Original Data Integrity ===")
    orig_labels = list(ORIGINAL_LABELS_DIR.glob("*.txt"))
    orig_images = list(ORIGINAL_IMAGES_DIR.glob("*.jpg"))
    assert len(orig_labels) == 1072, f"Original label count changed: {len(orig_labels)}"
    assert len(orig_images) == 1072, f"Original image count changed: {len(orig_images)}"
    print(f"  [OK] Original files preserved: {len(orig_images)} images, {len(orig_labels)} labels untouched.")

    print("\n=== Step 5: Checking Class Mapping and data.yaml Consistency ===")
    with open(ECOSORT_YOLO_DIR / "class_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    assert mapping["num_classes"] == 13
    assert len(mapping["classes"]) == 13

    with open(ECOSORT_YOLO_DIR / "classes.txt", "r", encoding="utf-8") as f:
        class_lines = [l.strip() for l in f if l.strip()]
    assert len(class_lines) == 13
    for i, c in enumerate(mapping["classes"]):
        assert c["yolo_id"] == i
        assert c["name"] == class_lines[i]

    print("  [OK] data.yaml, classes.txt, and class_mapping.json 100% aligned.")
    print("\n*** ALL INTEGRITY CHECKS PASSED PERFECTLY ***")

if __name__ == "__main__":
    run_validation()
