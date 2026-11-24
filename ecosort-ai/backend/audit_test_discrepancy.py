r"""Audit script to resolve test annotation discrepancy and verify ground truth counts.
"""

import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO")
SPLITS_DIR = BASE_DIR / "splits" / "verified_cc_only"
PROCESSED_LABELS_DIR = BASE_DIR / "processed" / "labels"
PROCESSED_IMAGES_DIR = BASE_DIR / "processed" / "images"
ANNOTATIONS_FILE = BASE_DIR / "annotations" / "annotations.json"
ECOSORT_YOLO_DIR = BASE_DIR / "ecosort_yolo"

def audit():
    # 1. Load split files
    splits = {}
    split_ids = {}
    for s in ["train", "val", "test"]:
        with open(SPLITS_DIR / f"{s}.txt", "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f if line.strip()]
        splits[s] = paths
        split_ids[s] = [int(Path(p).stem) for p in paths]
        print(f"{s} split: {len(paths)} images")

    all_ids = split_ids["train"] + split_ids["val"] + split_ids["test"]
    assert len(all_ids) == 319

    # 2. Load annotations.json
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        anno_data = json.load(f)

    categories = {c["id"]: c for c in anno_data["categories"]}
    img_to_annos = defaultdict(list)
    for a in anno_data["annotations"]:
        img_to_annos[a["image_id"]].append(a)

    print(f"Total annotations in annotations.json: {len(anno_data['annotations'])}")

    # 3. Count annotations from annotations.json for verified subset
    ann_json_counts = {"train": 0, "val": 0, "test": 0}
    ann_json_class_counts = {s: defaultdict(int) for s in ["train", "val", "test"]}

    for s in ["train", "val", "test"]:
        for iid in split_ids[s]:
            annos = img_to_annos[iid]
            ann_json_counts[s] += len(annos)
            for a in annos:
                ann_json_class_counts[s][a["category_id"]] += 1

    print(f"Annotations.json counts for verified subset:")
    print(f"  Train: {ann_json_counts['train']}")
    print(f"  Val:   {ann_json_counts['val']}")
    print(f"  Test:  {ann_json_counts['test']}")
    print(f"  Total: {sum(ann_json_counts.values())}")

    # 4. Count annotations directly from processed label files on disk
    disk_counts = {"train": 0, "val": 0, "test": 0}
    disk_class_counts = {s: defaultdict(int) for s in ["train", "val", "test"]}
    disk_distinct_imgs = {s: defaultdict(set) for s in ["train", "val", "test"]}

    for s in ["train", "val", "test"]:
        for iid in split_ids[s]:
            lbl_file = PROCESSED_LABELS_DIR / f"{iid:06d}.txt"
            if not lbl_file.exists():
                print(f"Missing label file: {lbl_file}")
                continue
            with open(lbl_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            disk_counts[s] += len(lines)
            for line in lines:
                parts = line.split()
                cid = int(parts[0])
                disk_class_counts[s][cid] += 1
                disk_distinct_imgs[s][cid].add(iid)

    print(f"\nProcessed label files (.txt) counts on disk:")
    print(f"  Train: {disk_counts['train']}")
    print(f"  Val:   {disk_counts['val']}")
    print(f"  Test:  {disk_counts['test']}")
    print(f"  Total: {sum(disk_counts.values())}")

    # Check for discrepancies between annotations.json and processed txt
    for s in ["train", "val", "test"]:
        assert ann_json_counts[s] == disk_counts[s], f"Discrepancy in {s}: json={ann_json_counts[s]}, disk={disk_counts[s]}"

    # 5. Let's see per-class breakdown for each split
    print("\n--- Detailed per-class breakdown ---")
    total_class_counts = {}
    for cid in range(60):
        tr = disk_class_counts["train"][cid]
        va = disk_class_counts["val"][cid]
        te = disk_class_counts["test"][cid]
        tot = tr + va + te
        imgs = len(disk_distinct_imgs["train"][cid] | disk_distinct_imgs["val"][cid] | disk_distinct_imgs["test"][cid])
        total_class_counts[cid] = (tot, imgs, tr, va, te)

    # Let's inspect where 104 vs 128 came from!
    print(f"\nTest split sum across all 60 classes: {sum(disk_class_counts['test'].values())}")
    print(f"Val split sum across all 60 classes: {sum(disk_class_counts['val'].values())}")
    print(f"Train split sum across all 60 classes: {sum(disk_class_counts['train'].values())}")

    # Check the remapped labels in ecosort_yolo/labels/
    remapped_counts = {"train": 0, "val": 0, "test": 0}
    remapped_class_counts = {s: defaultdict(int) for s in ["train", "val", "test"]}
    for s in ["train", "val", "test"]:
        lbl_dir = ECOSORT_YOLO_DIR / "labels" / s
        for lf in lbl_dir.glob("*.txt"):
            with open(lf, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            remapped_counts[s] += len(lines)
            for l in lines:
                cid = int(l.split()[0])
                remapped_class_counts[s][cid] += 1

    print(f"\nRemapped ecosort_yolo labels counts:")
    print(f"  Train: {remapped_counts['train']}")
    print(f"  Val:   {remapped_counts['val']}")
    print(f"  Test:  {remapped_counts['test']}")
    print(f"  Total: {sum(remapped_counts.values())}")

if __name__ == "__main__":
    audit()

def check_groups():
    dist = json.load(open(BASE_DIR / "manifests" / "verified_subset_class_distribution.json", encoding="utf-8"))
    groups = defaultdict(lambda: {"count": 0, "total": 0, "train": 0, "val": 0, "test": 0, "classes": []})
    for c in dist["classes"]:
        g = c["viability_group"]
        groups[g]["count"] += 1
        groups[g]["total"] += c["total_annotations"]
        groups[g]["train"] += c["train_annotations"]
        groups[g]["val"] += c["val_annotations"]
        groups[g]["test"] += c["test_annotations"]
        groups[g]["classes"].append((c["class_id"], c["category_name"], c["total_annotations"], c["train_annotations"], c["val_annotations"], c["test_annotations"]))

    print(f"\nGroup breakdown from verified_subset_class_distribution.json:")
    tot_train = 0
    tot_val = 0
    tot_test = 0
    tot_all = 0
    for g, data in groups.items():
        print(f"  {g} ({data['count']} classes): Total={data['total']}, Train={data['train']}, Val={data['val']}, Test={data['test']}")
        tot_train += data['train']
        tot_val += data['val']
        tot_test += data['test']
        tot_all += data['total']
    print(f"  SUM: Total={tot_all}, Train={tot_train}, Val={tot_val}, Test={tot_test}")

    print("\nClasses in Fewer than 5 Examples:")
    for cid, name, tot, tr, va, te in groups["Fewer than 5 Examples"]["classes"]:
        print(f"    ID {cid:2d}: {name:<25} tot={tot:2d}, tr={tr:2d}, va={va:2d}, te={te:2d}")

    print("\nClasses in 5 to 14 Examples:")
    for cid, name, tot, tr, va, te in groups["5 to 14 Examples"]["classes"]:
        print(f"    ID {cid:2d}: {name:<25} tot={tot:2d}, tr={tr:2d}, va={va:2d}, te={te:2d}")

    print("\nClasses in High-Clutter / Excluded Classes:")
    for cid, name, tot, tr, va, te in groups["High-Clutter / Excluded Classes"]["classes"]:
        print(f"    ID {cid:2d}: {name:<25} tot={tot:2d}, tr={tr:2d}, va={va:2d}, te={te:2d}")

    print("\nClasses in Candidate Classes for Baseline Experimentation:")
    for cid, name, tot, tr, va, te in groups["Candidate Classes for Baseline Experimentation"]["classes"]:
        print(f"    ID {cid:2d}: {name:<25} tot={tot:2d}, tr={tr:2d}, va={va:2d}, te={te:2d}")

if __name__ == "__main__":
    check_groups()

def check_zero():
    dist = json.load(open(BASE_DIR / "manifests" / "verified_subset_class_distribution.json", encoding="utf-8"))
    zero_classes = [c for c in dist["classes"] if c["viability_group"] == "Zero Examples"]
    print(f"\nZero classes count: {len(zero_classes)}")
    for c in zero_classes:
        print(f"  ID {c['class_id']:2d}: {c['category_name']}")

if __name__ == "__main__":
    check_zero()
