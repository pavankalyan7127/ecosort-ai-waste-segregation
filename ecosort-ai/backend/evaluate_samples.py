"""Validation script for real YOLO11n inference on test samples."""

import json
import os
import sys
import time
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from app.vision.detector import WasteDetector

detector = WasteDetector()
sample_dir = Path(r"D:\EcoSort-Resources\experiments\test_samples")

test_files = [
    "sample1_plastic_bottle.jpg",
    "sample2_bananas_fruit.jpg",
    "sample3_person_chair_bottle.jpg",
]

print("=" * 70)
print("ECOSORT AI — REAL INFERENCE VALIDATION ON SAMPLE IMAGES")
print("=" * 70)

all_results = []

for fname in test_files:
    fpath = sample_dir / fname
    if not fpath.exists():
        print(f"Error: {fpath} does not exist!")
        continue

    with open(fpath, "rb") as f:
        img_bytes = f.read()

    t0 = time.perf_counter()
    result = detector.detect(img_bytes, confidence_threshold=0.25)
    t1 = time.perf_counter()

    result["file_name"] = fname
    result["file_size_bytes"] = len(img_bytes)
    result["total_wall_time_ms"] = round((t1 - t0) * 1000, 2)
    all_results.append(result)

    print(f"\nFILE: {fname} ({len(img_bytes)} bytes)")
    print(f"Pipeline Status: {result['pipeline_status']} | Model: {result['target_model']}")
    print(f"Image Dimensions: {result['image_dimensions']}")
    print(f"Total Objects Detected: {result['total_detected']}")
    print(f"Total Waste-Mapped: {result['total_waste_mapped']}")
    print(f"Stream Counts: {result['stream_counts']}")
    print(f"YOLO Inference Latency: {result['inference_latency_ms']} ms | Wall time: {result['total_wall_time_ms']} ms")
    print("Detected Items:")
    for idx, item in enumerate(result["detected_items"], 1):
        status_label = "[MAPPED]" if item["is_waste_mapped"] else "[UNSUPPORTED]"
        print(f"  {idx}. {status_label} '{item['detected_label']}' (Conf: {item['confidence']*100:.1f}%)")
        print(f"     Stream: {item['primary_stream']} | Bin Color: {item['bin_color']}")
        print(f"     Bounding Box (normalized): {item['bounding_box']}")
        if item["is_waste_mapped"]:
            print(f"     Statutory Item: {item['item_name']} (ID: {item['class_id']})")
            print(f"     Verification Status: {item['verification_status']}")
            print(f"     Disposal Directive: {item['disposal_guidance'][:80]}...")
            if item.get("source_references"):
                src = item["source_references"][0]
                print(f"     Legal Source: {src.get('document_title')} - Rule {src.get('rule_or_section')} (Page {src.get('pdf_page')})")
        else:
            print(f"     Mapping Notes: {item['mapping_notes']}")

out_json = backend_dir.parent / "docs" / "benchmarks" / "inference_validation_results.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 70)
print(f"Complete inference validation results saved to {out_json}")
print("=" * 70)
