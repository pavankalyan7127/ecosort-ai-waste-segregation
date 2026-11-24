"""Complete script for TACO dataset licensing audit, annotation reconciliation,

duplicate clustering, class distribution analysis, and train/val/test splitting.
Target: D:\EcoSort-Resources\datasets\TACO\
"""

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO")
ORIGINAL_DIR = BASE_DIR / "original"
PROCESSED_DIR = BASE_DIR / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
LABELS_DIR = PROCESSED_DIR / "labels"
MANIFESTS_DIR = BASE_DIR / "manifests"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
SPLITS_DIR = BASE_DIR / "splits"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def fix_coordinate_rounding_edges():
    """Ensure coordinates in YOLO txt labels are strictly bounded in [0.0, 1.0]."""
    fixed_count = 0
    for l_file in LABELS_DIR.glob("*.txt"):
        new_lines = []
        changed = False
        for line in open(l_file, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            # Compute corners
            x1 = max(0.0, xc - w / 2.0)
            y1 = max(0.0, yc - h / 2.0)
            x2 = min(1.0, xc + w / 2.0)
            y2 = min(1.0, yc + h / 2.0)

            new_w = x2 - x1
            new_h = y2 - y1
            new_xc = x1 + new_w / 2.0
            new_yc = y1 + new_h / 2.0

            if abs(new_xc - xc) > 1e-6 or abs(new_yc - yc) > 1e-6 or abs(new_w - w) > 1e-6 or abs(new_h - h) > 1e-6:
                changed = True
                fixed_count += 1

            new_lines.append(f"{cls_id} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}")

        if changed:
            with open(l_file, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")

    logger.info(f"Fixed {fixed_count} micro-rounded bounding box edges to enforce strict [0.0, 1.0] boundary containment.")


def run_licensing_audit(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Produce corrected, rigorous image-level licensing audit."""
    images = data["images"]
    audit_rows = []

    for img in images:
        img_id = img["id"]
        file_name = img["file_name"]
        raw_lic = img.get("license")
        flickr_url = img.get("flickr_url") or ""
        flickr_640_url = img.get("flickr_640_url") or ""

        # Corrected categorization:
        # 1. Verified: Explicit CC tag
        # 2. Conditional: OpenLitterMap ODbL
        # 3. Unclear / requires manual verification: raw_lic is null/None
        if raw_lic == "CC":
            rights_tier = "Verified usage rights"
            lic_category = "Creative Commons (Flickr CC BY / CC BY-SA)"
            commercial_ok = True
            redistribute_ok = True
            share_alike = False
            attribution_req = True
            restrictions = "Standard CC terms: attribution to creator required. Permissible for AI training."
        elif raw_lic == "ODBL (c) OpenLitterMap & Contributors" or "olm-s3" in flickr_url:
            rights_tier = "Conditional usage rights"
            lic_category = "Open Database License (ODbL) v1.0"
            commercial_ok = True
            redistribute_ok = True
            share_alike = True
            attribution_req = True
            restrictions = "ODbL copyleft: database derivatives require Share-Alike terms; OpenLitterMap attribution mandatory."
        else:
            rights_tier = "Unclear / requires manual verification"
            lic_category = "Unspecified (TACO Web Upload / Unverified Flickr)"
            commercial_ok = False
            redistribute_ok = False
            share_alike = False
            attribution_req = True
            restrictions = "License field null in metadata; individual photographer rights unverified. Safe for internal evaluation only."

        audit_rows.append({
            "image_id": img_id,
            "file_name": file_name,
            "rights_classification": rights_tier,
            "standardized_license": lic_category,
            "raw_license_in_metadata": str(raw_lic),
            "commercial_allowed": commercial_ok,
            "redistribution_allowed": redistribute_ok,
            "share_alike_required": share_alike,
            "attribution_required": attribution_req,
            "source_url": flickr_url,
            "thumbnail_url": flickr_640_url,
            "legal_restriction_notes": restrictions,
        })

    # Save corrected audit CSV and JSON
    out_csv = MANIFESTS_DIR / "licensing_audit_corrected.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    out_json = MANIFESTS_DIR / "licensing_audit_corrected.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_rows, f, indent=2)

    logger.info(f"Corrected licensing audit written to {out_csv} and {out_json}")
    return audit_rows


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Calculate 64-bit difference hash (dHash)."""
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def compute_duplicate_clusters(image_ids: List[int]) -> Dict[int, List[int]]:
    """Compute exact cryptographic and perceptual duplicate clusters."""
    hashes = {}
    for img_id in image_ids:
        p = IMAGES_DIR / f"{img_id:06d}.jpg"
        with open(p, "rb") as f:
            bytes_data = f.read()
        md5_val = hashlib.md5(bytes_data).hexdigest()
        with Image.open(p) as img:
            phash_val = dhash(img)
        hashes[img_id] = {"md5": md5_val, "dhash": phash_val}

    parent = {img_id: img_id for img_id in image_ids}

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[ri] = rj

    # Group by exact MD5 or perceptual Hamming distance <= 4
    for i in range(len(image_ids)):
        id1 = image_ids[i]
        for j in range(i + 1, len(image_ids)):
            id2 = image_ids[j]
            if hashes[id1]["md5"] == hashes[id2]["md5"]:
                union(id1, id2)
            else:
                dist = bin(hashes[id1]["dhash"] ^ hashes[id2]["dhash"]).count("1")
                if dist <= 4:
                    union(id1, id2)

    clusters = {}
    for img_id in image_ids:
        root = find(img_id)
        clusters.setdefault(root, []).append(img_id)

    return clusters


def multilabel_stratified_split(
    clusters: Dict[int, List[int]],
    anns_by_img: Dict[int, List[Dict[str, Any]]],
    num_classes: int = 60,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Group-aware, multilabel-balanced split across train, val, and test partitions."""
    import random
    rng = random.Random(seed)

    # Compute class histogram for each cluster
    cluster_list = list(clusters.values())
    rng.shuffle(cluster_list)

    total_images = sum(len(c) for c in cluster_list)
    target_train = int(round(total_images * ratios[0]))
    target_val = int(round(total_images * ratios[1]))
    target_test = total_images - target_train - target_val

    cluster_profiles = []
    for c in cluster_list:
        class_counts = [0] * num_classes
        for img_id in c:
            for ann in anns_by_img.get(img_id, []):
                cid = ann["category_id"]
                if 0 <= cid < num_classes:
                    class_counts[cid] += 1
        cluster_profiles.append({"members": c, "counts": class_counts, "size": len(c)})

    # Sort clusters by presence of rare classes first (heuristics for multilabel balancing)
    global_class_counts = [0] * num_classes
    for cp in cluster_profiles:
        for cid in range(num_classes):
            global_class_counts[cid] += cp["counts"][cid]

    def cluster_rarity_score(cp):
        # Minimum frequency of any class present in the cluster
        present_freqs = [global_class_counts[cid] for cid in range(num_classes) if cp["counts"][cid] > 0]
        return min(present_freqs) if present_freqs else 999999

    cluster_profiles.sort(key=cluster_rarity_score)

    train_members, val_members, test_members = [], [], []
    train_class_counts = [0] * num_classes
    val_class_counts = [0] * num_classes
    test_class_counts = [0] * num_classes

    for cp in cluster_profiles:
        members = cp["members"]
        size = cp["size"]

        # If train is not full and has highest need
        curr_train_len = len(train_members)
        curr_val_len = len(val_members)
        curr_test_len = len(test_members)

        # Decide split placement
        scores = []
        # Candidate 0: train
        can_train = curr_train_len + size <= target_train + 15
        can_val = curr_val_len + size <= target_val + 5
        can_test = curr_test_len + size <= target_test + 5

        if can_val and curr_val_len / max(1, target_val) <= curr_train_len / max(1, target_train) and curr_val_len / max(1, target_val) <= curr_test_len / max(1, target_test):
            val_members.extend(members)
            for cid in range(num_classes):
                val_class_counts[cid] += cp["counts"][cid]
        elif can_test and curr_test_len / max(1, target_test) <= curr_train_len / max(1, target_train):
            test_members.extend(members)
            for cid in range(num_classes):
                test_class_counts[cid] += cp["counts"][cid]
        else:
            train_members.extend(members)
            for cid in range(num_classes):
                train_class_counts[cid] += cp["counts"][cid]

    return sorted(train_members), sorted(val_members), sorted(test_members)


def main():
    logger.info("Starting TACO Dataset Task 5.2.2 Audit & Split Preparation...")

    # Step 1: Fix bounding box rounding edges in YOLO labels
    fix_coordinate_rounding_edges()

    # Step 2: Load metadata
    with open(ANNOTATIONS_DIR / "annotations.json", encoding="utf-8") as f:
        data = json.load(f)

    categories = {c["id"]: c for c in data["categories"]}
    annotations = data["annotations"]

    # Step 3: Licensing Audit
    audit_manifest = run_licensing_audit(data)
    audit_by_id = {row["image_id"]: row for row in audit_manifest}

    # Step 4: Identify all processed images on disk
    downloaded_ids = sorted([int(p.stem) for p in IMAGES_DIR.glob("*.jpg")])
    anns_by_img = {}
    for ann in annotations:
        if ann["image_id"] in downloaded_ids:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Step 5: Duplicate Clustering
    logger.info(f"Computing duplicate clusters across {len(downloaded_ids)} images...")
    clusters = compute_duplicate_clusters(downloaded_ids)
    multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    logger.info(f"Found {len(multi_clusters)} duplicate/near-duplicate clusters comprising {sum(len(v) for v in multi_clusters.values())} images.")

    # Step 6: Generate Splits
    # Split Configuration A: Full Available Dataset (1,072 images)
    train_ids, val_ids, test_ids = multilabel_stratified_split(clusters, anns_by_img, num_classes=60, ratios=(0.8, 0.1, 0.1), seed=SEED)

    # Verify zero data leakage
    s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
    assert len(s_train & s_val) == 0, "Train-Val leakage detected!"
    assert len(s_train & s_test) == 0, "Train-Test leakage detected!"
    assert len(s_val & s_test) == 0, "Val-Test leakage detected!"
    assert len(s_train | s_val | s_test) == len(downloaded_ids), "Missing images in split!"

    # Verify cluster integrity (all cluster members in same split)
    for root, members in multi_clusters.items():
        in_splits = set()
        for m in members:
            if m in s_train: in_splits.add("train")
            elif m in s_val: in_splits.add("val")
            elif m in s_test: in_splits.add("test")
        assert len(in_splits) == 1, f"Cluster {members} split across multiple partitions: {in_splits}!"

    logger.info(f"Full Split Generated: Train={len(train_ids)} ({len(train_ids)/len(downloaded_ids)*100:.1f}%), Val={len(val_ids)} ({len(val_ids)/len(downloaded_ids)*100:.1f}%), Test={len(test_ids)} ({len(test_ids)/len(downloaded_ids)*100:.1f}%)")

    # Write train.txt, val.txt, test.txt
    for split_name, id_list in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        txt_path = SPLITS_DIR / f"{split_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for img_id in id_list:
                img_path = IMAGES_DIR / f"{img_id:06d}.jpg"
                f.write(f"{img_path.as_posix()}\n")

    # Step 7: Sub-split for Verified CC Rights Only (319 images)
    verified_ids = [img_id for img_id in downloaded_ids if audit_by_id[img_id]["rights_classification"] == "Verified usage rights"]
    v_clusters = compute_duplicate_clusters(verified_ids)
    v_train, v_val, v_test = multilabel_stratified_split(v_clusters, anns_by_img, num_classes=60, ratios=(0.8, 0.1, 0.1), seed=SEED)

    v_splits_dir = SPLITS_DIR / "verified_cc_only"
    v_splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, id_list in [("train", v_train), ("val", v_val), ("test", v_test)]:
        with open(v_splits_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
            for img_id in id_list:
                f.write(f"{(IMAGES_DIR / f'{img_id:06d}.jpg').as_posix()}\n")

    logger.info(f"Verified CC Sub-split Generated: Train={len(v_train)}, Val={len(v_val)}, Test={len(v_test)} (Total: {len(verified_ids)} images).")

    # Step 8: Compute Class Distribution across splits
    def compute_distribution(id_list):
        counts = {cid: 0 for cid in range(60)}
        images_with_cls = {cid: set() for cid in range(60)}
        for img_id in id_list:
            for ann in anns_by_img.get(img_id, []):
                cid = ann["category_id"]
                if 0 <= cid < 60:
                    counts[cid] += 1
                    images_with_cls[cid].add(img_id)
        return {
            cid: {
                "category_name": categories[cid]["name"],
                "supercategory": categories[cid]["supercategory"],
                "annotation_count": counts[cid],
                "image_count": len(images_with_cls[cid]),
            }
            for cid in range(60)
        }

    full_dist = compute_distribution(downloaded_ids)
    train_dist = compute_distribution(train_ids)
    val_dist = compute_distribution(val_ids)
    test_dist = compute_distribution(test_ids)

    # Class distribution JSON
    dist_payload = {
        "dataset_name": "TACO (Option B 640x640 Processed)",
        "random_seed": SEED,
        "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "total_downloaded_images": len(downloaded_ids),
        "total_downloaded_annotations": sum(len(anns_by_img.get(i, [])) for i in downloaded_ids),
        "image_counts_by_split": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "annotation_counts_by_split": {
            "train": sum(train_dist[c]["annotation_count"] for c in range(60)),
            "val": sum(val_dist[c]["annotation_count"] for c in range(60)),
            "test": sum(test_dist[c]["annotation_count"] for c in range(60)),
        },
        "classes": {
            cid: {
                "taco_id": cid,
                "name": categories[cid]["name"],
                "supercategory": categories[cid]["supercategory"],
                "total_annotations": full_dist[cid]["annotation_count"],
                "total_images": full_dist[cid]["image_count"],
                "train_annotations": train_dist[cid]["annotation_count"],
                "val_annotations": val_dist[cid]["annotation_count"],
                "test_annotations": test_dist[cid]["annotation_count"],
                "statistically_evaluable": full_dist[cid]["annotation_count"] >= 15,
            }
            for cid in range(60)
        },
    }

    with open(SPLITS_DIR / "class_distribution.json", "w", encoding="utf-8") as f:
        json.dump(dist_payload, f, indent=2)

    # Split Manifest JSON
    split_manifest = {
        "metadata": {
            "random_seed": SEED,
            "split_date": "2026-11-05",
            "duplicate_grouping_enforced": True,
            "data_leakage_prevented": True,
        },
        "image_records": [
            {
                "image_id": img_id,
                "file_name": f"{img_id:06d}.jpg",
                "assigned_split": "train" if img_id in s_train else ("val" if img_id in s_val else "test"),
                "cluster_id": [k for k, v in clusters.items() if img_id in v][0],
                "licensing_status": audit_by_id[img_id]["rights_classification"],
                "standardized_license": audit_by_id[img_id]["standardized_license"],
                "annotations_count": len(anns_by_img.get(img_id, [])),
            }
            for img_id in downloaded_ids
        ],
    }

    with open(SPLITS_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    logger.info("Generated split_manifest.json and class_distribution.json successfully.")


if __name__ == "__main__":
    main()
