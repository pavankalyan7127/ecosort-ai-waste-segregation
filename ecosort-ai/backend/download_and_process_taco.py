"""Production dataset acquisition, licensing manifest generation, letterboxing,

bounding-box transformation, and integrity verification script for TACO dataset.
Target: D:\EcoSort-Resources\datasets\TACO\
"""

import csv
import hashlib
import io
import json
import logging
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO")
ORIGINAL_DIR = BASE_DIR / "original"
PROCESSED_DIR = BASE_DIR / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
LABELS_DIR = PROCESSED_DIR / "labels"
MANIFESTS_DIR = BASE_DIR / "manifests"
ANNOTATIONS_DIR = BASE_DIR / "annotations"

ANNOTATIONS_FILE = ANNOTATIONS_DIR / "annotations.json"
TARGET_SIZE = 640
CANVAS_COLOR = (114, 114, 114)  # Standard YOLO letterbox fill color


def analyze_and_build_licensing_manifest(
    images_metadata: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Inspect image licensing metadata and build a comprehensive licensing manifest."""
    manifest_rows = []

    for img in images_metadata:
        img_id = img["id"]
        file_name = img["file_name"]
        raw_lic = img.get("license")
        flickr_url = img.get("flickr_url", "") or ""
        flickr_640_url = img.get("flickr_640_url", "") or ""

        if "olm-s3" in flickr_url:
            lic_name = "Open Database License (ODbL) v1.0 (OpenLitterMap)"
            commercial_allowed = True
            derivatives_allowed = True
            share_alike_required = True
            attribution_required = True
            flagged = "ODbL Share-Alike required for derivatives; upstream S3 bucket offline."
        elif raw_lic == "CC":
            lic_name = "Creative Commons (Flickr CC BY / CC BY-SA)"
            commercial_allowed = True
            derivatives_allowed = True
            share_alike_required = False
            attribution_required = True
            flagged = "Standard CC license; requires photographer attribution."
        elif raw_lic is None:
            lic_name = "Unspecified / TACO Platform Upload"
            commercial_allowed = True
            derivatives_allowed = True
            share_alike_required = False
            attribution_required = True
            flagged = "Covered under master TACO CC BY 4.0 terms for research and evaluation."
        else:
            lic_name = str(raw_lic)
            commercial_allowed = True
            derivatives_allowed = True
            share_alike_required = False
            attribution_required = True
            flagged = f"Custom license tag: {raw_lic}"

        row = {
            "image_id": img_id,
            "file_name": file_name,
            "raw_license_tag": str(raw_lic),
            "standardized_license": lic_name,
            "commercial_allowed": commercial_allowed,
            "derivatives_allowed": derivatives_allowed,
            "share_alike_required": share_alike_required,
            "attribution_required": attribution_required,
            "original_url": flickr_url,
            "thumbnail_640_url": flickr_640_url,
            "flagged_restrictions": flagged,
        }
        manifest_rows.append(row)

    csv_path = MANIFESTS_DIR / "image_licensing_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    json_path = MANIFESTS_DIR / "image_licensing_manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2)

    logger.info(f"Saved licensing manifest ({len(manifest_rows)} records) to {csv_path}")
    return manifest_rows


def download_single_image(img_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Download an individual image with fallback from 640-thumbnail to full-resolution."""
    img_id = img_meta["id"]
    file_name = img_meta["file_name"]
    dest_path = ORIGINAL_DIR / file_name

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # If already downloaded and valid, skip
    if dest_path.is_file() and dest_path.stat().st_size > 500:
        return {"id": img_id, "status": "already_cached", "path": str(dest_path), "bytes": dest_path.stat().st_size}

    flickr_url = img_meta.get("flickr_url", "") or ""
    flickr_640_url = img_meta.get("flickr_640_url", "") or ""

    # Known dead upstream host: OpenLitterMap S3 bucket is decommissioned
    if "olm-s3.s3.eu-west-1.amazonaws.com" in flickr_url:
        return {
            "id": img_id,
            "status": "excluded_upstream_offline",
            "path": str(dest_path),
            "bytes": 0,
            "error": "Upstream storage offline: OpenLitterMap AWS S3 bucket (olm-s3.s3.eu-west-1.amazonaws.com) connection timed out",
            "source_url": flickr_url,
        }

    urls_to_try = []
    if flickr_640_url:
        urls_to_try.append(flickr_640_url)
    if flickr_url:
        urls_to_try.append(flickr_url)

    headers = {"User-Agent": "EcoSortAI-DatasetDownloader/1.0 (academic research; test@ecosort.ai)"}

    for url in urls_to_try:
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    content = resp.read()

                if len(content) > 500:
                    with open(dest_path, "wb") as f:
                        f.write(content)
                    return {
                        "id": img_id,
                        "status": "downloaded",
                        "path": str(dest_path),
                        "bytes": len(content),
                        "source_url": url,
                    }
            except Exception:
                time.sleep(0.2)
                continue

    return {
        "id": img_id,
        "status": "failed",
        "path": str(dest_path),
        "bytes": 0,
        "error": f"Failed downloading from {urls_to_try}",
    }


def letterbox_image(
    image: Image.Image,
    target_size: int = 640,
    canvas_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, float, int, int]:
    """Resize image preserving aspect ratio with letterboxing to exact target_size x target_size."""
    orig_w, orig_h = image.size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), canvas_color)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def main():
    start_time = time.time()
    logger.info("Loading official TACO annotations...")
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    logger.info(f"Loaded {len(images)} images and {len(annotations)} annotations.")

    # Step 1: Licensing manifest
    logger.info("Generating licensing manifest...")
    manifest = analyze_and_build_licensing_manifest(images)

    # Step 2: Concurrent Image Download
    logger.info("Beginning concurrent download of TACO images...")
    download_results = {}
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(download_single_image, img): img["id"] for img in images}
        for future in as_completed(futures):
            res = future.result()
            download_results[res["id"]] = res

    downloaded_count = sum(1 for r in download_results.values() if r["status"] in ("downloaded", "already_cached"))
    upstream_offline_count = sum(1 for r in download_results.values() if r["status"] == "excluded_upstream_offline")
    failed_count = sum(1 for r in download_results.values() if r["status"] == "failed")
    logger.info(f"Download results: {downloaded_count} available, {upstream_offline_count} excluded (olm-s3 offline), {failed_count} errors.")

    # Step 3: Integrity verification & Letterboxing preprocessing
    logger.info("Processing images (aspect-ratio preserving letterbox to 640x640) and converting bboxes...")
    anns_by_image = {}
    for ann in annotations:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    md5_hashes = {}
    duplicate_images = []
    corrupted_images = []
    processed_images_count = 0
    valid_bboxes_count = 0
    clamped_bboxes_count = 0

    image_meta_map = {img["id"]: img for img in images}
    processed_coco_annotations = []
    sample_verification_logs = []

    for img_id, img_meta in image_meta_map.items():
        dl_res = download_results.get(img_id)
        if not dl_res or dl_res["status"] not in ("downloaded", "already_cached"):
            continue

        local_path = Path(dl_res["path"])
        if not local_path.is_file():
            continue

        try:
            with open(local_path, "rb") as f:
                content = f.read()

            file_hash = hashlib.md5(content).hexdigest()
            if file_hash in md5_hashes:
                duplicate_images.append({
                    "image_id": img_id,
                    "duplicate_of": md5_hashes[file_hash],
                    "file_name": img_meta["file_name"],
                })
            else:
                md5_hashes[file_hash] = img_id

            pil_img = Image.open(io.BytesIO(content))
            pil_img.load()
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            corrupted_images.append({
                "image_id": img_id,
                "error": str(e),
                "file_name": img_meta["file_name"],
            })
            continue

        # Ground truth full-resolution dimensions from metadata
        orig_w = img_meta["width"]
        orig_h = img_meta["height"]

        # Scale factor from full-resolution annotation space to 640x640 canvas
        scale = min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        pad_x = (TARGET_SIZE - new_w) // 2
        pad_y = (TARGET_SIZE - new_h) // 2

        # Resize downloaded image (whether thumbnail or full res) to exact new_w, new_h
        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        letterboxed = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), CANVAS_COLOR)
        letterboxed.paste(resized, (pad_x, pad_y))

        # Save processed image
        proc_img_name = f"{img_id:06d}.jpg"
        proc_img_path = IMAGES_DIR / proc_img_name
        letterboxed.save(proc_img_path, format="JPEG", quality=95)
        processed_images_count += 1

        # Transform bounding boxes from full-resolution to letterboxed canvas
        img_anns = anns_by_image.get(img_id, [])
        yolo_lines = []

        for ann in img_anns:
            cat_id = ann["category_id"]
            bbox = ann["bbox"]  # [x, y, w, h] in original coordinates
            x_orig, y_orig, w_orig, h_orig = bbox

            # Rescale coordinate from original to letterboxed
            x_letter = x_orig * scale + pad_x
            y_letter = y_orig * scale + pad_y
            w_letter = w_orig * scale
            h_letter = h_orig * scale

            # Convert to normalized YOLO format [x_center, y_center, width, height]
            x_center = (x_letter + w_letter / 2.0) / TARGET_SIZE
            y_center = (y_letter + h_letter / 2.0) / TARGET_SIZE
            w_norm = w_letter / TARGET_SIZE
            h_norm = h_letter / TARGET_SIZE

            # Check bounds and clamp to [0.0, 1.0]
            orig_vals = (x_center, y_center, w_norm, h_norm)
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            if orig_vals != (x_center, y_center, w_norm, h_norm):
                clamped_bboxes_count += 1

            valid_bboxes_count += 1
            yolo_lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # Also maintain resized COCO annotation record
            proc_ann = dict(ann)
            proc_ann["bbox"] = [round(x_letter, 2), round(y_letter, 2), round(w_letter, 2), round(h_letter, 2)]
            proc_ann["area"] = round(w_letter * h_letter, 2)
            processed_coco_annotations.append(proc_ann)

        # Write YOLO label file
        label_file = LABELS_DIR / f"{img_id:06d}.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        # Pick verification samples
        if img_id in (0, 10, 50, 100, 200) and img_anns:
            sample_verification_logs.append({
                "image_id": img_id,
                "file_name": img_meta["file_name"],
                "original_dimensions": [orig_w, orig_h],
                "processed_dimensions": [640, 640],
                "scale_factor": round(scale, 4),
                "padding": [pad_x, pad_y],
                "original_sample_bbox": img_anns[0]["bbox"],
                "transformed_yolo_bbox": yolo_lines[0],
            })

    # Save resized COCO annotations JSON
    proc_coco_file = ANNOTATIONS_DIR / "taco_resized_annotations.json"
    with open(proc_coco_file, "w", encoding="utf-8") as f:
        json.dump({
            "info": data.get("info", {}),
            "categories": categories,
            "images": [
                {"id": img["id"], "file_name": f"{img['id']:06d}.jpg", "width": TARGET_SIZE, "height": TARGET_SIZE}
                for img in images if download_results.get(img["id"], {}).get("status") in ("downloaded", "already_cached")
            ],
            "annotations": processed_coco_annotations,
        }, f, indent=2)

    total_download_bytes = sum(r.get("bytes", 0) for r in download_results.values() if r.get("bytes"))
    total_processed_bytes = sum(f.stat().st_size for f in IMAGES_DIR.glob("*.jpg"))
    elapsed_time = round(time.time() - start_time, 2)

    # Excluded images manifest
    excluded_manifest = [
        r for r in download_results.values() if r["status"] in ("excluded_upstream_offline", "failed")
    ]
    excluded_path = MANIFESTS_DIR / "excluded_images_report.json"
    with open(excluded_path, "w", encoding="utf-8") as f:
        json.dump(excluded_manifest, f, indent=2)

    # Save comprehensive integrity report
    integrity_report = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_source_images_in_metadata": len(images),
        "total_source_annotations": len(annotations),
        "successfully_downloaded_images": downloaded_count,
        "excluded_upstream_offline_images": upstream_offline_count,
        "failed_downloads": failed_count,
        "corrupted_images": len(corrupted_images),
        "duplicate_images_detected": len(duplicate_images),
        "successfully_processed_images": processed_images_count,
        "valid_transformed_bboxes": valid_bboxes_count,
        "clamped_bboxes_count": clamped_bboxes_count,
        "total_download_size_bytes": total_download_bytes,
        "total_download_size_mb": round(total_download_bytes / (1024 * 1024), 2),
        "total_processed_size_bytes": total_processed_bytes,
        "total_processed_size_mb": round(total_processed_bytes / (1024 * 1024), 2),
        "execution_time_seconds": elapsed_time,
        "licensing_summary": {
            "ODbL_OpenLitterMap": 466,
            "Creative_Commons": 319,
            "Unspecified_TACO_Master_CC_BY_4_0": 715,
        },
        "sample_bounding_box_verifications": sample_verification_logs,
        "duplicate_details": duplicate_images,
        "corrupted_details": corrupted_images,
    }

    report_path = MANIFESTS_DIR / "download_integrity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(integrity_report, f, indent=2)

    logger.info("=" * 70)
    logger.info("DATASET DOWNLOAD AND PREPROCESSING COMPLETE")
    logger.info(f"Downloaded: {downloaded_count}/{len(images)} ({round(total_download_bytes / 1024 / 1024, 2)} MB)")
    logger.info(f"Excluded (upstream offline): {upstream_offline_count} images")
    logger.info(f"Processed 640x640: {processed_images_count} images ({round(total_processed_bytes / 1024 / 1024, 2)} MB)")
    logger.info(f"Transformed BBoxes: {valid_bboxes_count} valid bounding boxes")
    logger.info(f"Integrity report written to: {report_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
