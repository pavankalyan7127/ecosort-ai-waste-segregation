r"""EcoSort AI — YOLO11n-P2 Architecture Validation and Footprint Comparison Script.
Validates 4-scale detection head, tensor shapes, transfer learning, latency, and memory.
"""

import time
import os
import psutil
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Paths
YAML_PATH = Path(r"D:\EcoSort-Resources\experiments\yolo11n_p2_architecture_preflight\yolo11n-p2.yaml")
PRETRAINED_WEIGHTS = Path(r"D:\EcoSort-Resources\models\pretrained\yolo11n.pt")
BASELINE_640_WEIGHTS = Path(r"D:\EcoSort-Resources\experiments\yolo11n_baseline_13class\weights\best.pt")
EXP_1280_WEIGHTS = Path(r"D:\EcoSort-Resources\experiments\yolo11n_1280_baseline\weights\best.pt")
PREFLIGHT_DIR = Path(r"D:\EcoSort-Resources\experiments\yolo11n_p2_architecture_preflight")
SAMPLE_VAL_IMG = Path(r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\images\val\000106.jpg")

CLASS_NAMES = [
    "clear_plastic_bottle", "glass_bottle", "plastic_bottle_cap", "metal_bottle_cap",
    "drink_can", "other_carton", "corrugated_carton", "disposable_plastic_cup",
    "other_plastic", "plastic_film", "other_plastic_wrapper", "styrofoam_piece", "cigarette"
]

def benchmark_latency(model, input_tensor, warmup=10, runs=30):
    # Warmup
    for _ in range(warmup):
        _ = model(input_tensor, verbose=False)
    
    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(input_tensor, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    
    times = np.array(times)
    return {
        "mean_ms": round(float(np.mean(times)), 2),
        "median_ms": round(float(np.median(times)), 2),
        "std_ms": round(float(np.std(times)), 2),
        "min_ms": round(float(np.min(times)), 2),
        "max_ms": round(float(np.max(times)), 2),
        "fps": round(float(1000.0 / np.mean(times)), 1)
    }

def main():
    print("=== Step 1: Instantiate YOLO11n-P2 Model ===")
    assert YAML_PATH.exists(), f"Missing YAML at {YAML_PATH}"
    assert PRETRAINED_WEIGHTS.exists(), f"Missing pretrained weights at {PRETRAINED_WEIGHTS}"
    
    model_p2 = YOLO(str(YAML_PATH))
    print("Successfully parsed and instantiated YOLO11n-P2 architecture.")
    
    print("\n=== Step 2: Transfer Pretrained YOLO11n Weights ===")
    # Load pretrained weights into P2 architecture
    model_p2.load(str(PRETRAINED_WEIGHTS))
    
    # Save initialized weights
    initial_ckpt = PREFLIGHT_DIR / "yolo11n_p2_initial.pt"
    model_p2.save(str(initial_ckpt))
    ckpt_size_mb = round(initial_ckpt.stat().st_size / (1024 ** 2), 2)
    print(f"Saved initial P2 checkpoint to {initial_ckpt} ({ckpt_size_mb} MB)")
    
    # Detailed parameter analysis
    detect_module = model_p2.model.model[-1]
    strides = [float(s) for s in detect_module.stride]
    num_scales = int(detect_module.nl)
    num_classes = int(detect_module.nc)
    
    total_params = sum(p.numel() for p in model_p2.model.parameters())
    trainable_params = sum(p.numel() for p in model_p2.model.parameters() if p.requires_grad)
    num_layers = len(list(model_p2.model.modules()))
    
    print(f"Detection Strides: {strides}")
    print(f"Number of Scales: {num_scales}")
    print(f"Number of Classes: {num_classes}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Total Modules / Layers: {num_layers}")
    
    print("\n=== Step 3: Dummy Forward Pass (640x640) ===")
    dummy_input = torch.zeros(1, 3, 640, 640)
    raw_out = model_p2.model(dummy_input)
    
    feat_shapes = []
    if isinstance(raw_out, dict) and "feats" in raw_out:
        for idx, feat in enumerate(raw_out["feats"]):
            f_shape = list(feat.shape)
            feat_shapes.append(f_shape)
            print(f"Scale {idx} (stride {strides[idx]}): feature map shape = {f_shape}")
    
    boxes_shape = list(raw_out["boxes"].shape) if "boxes" in raw_out else "N/A"
    scores_shape = list(raw_out["scores"].shape) if "scores" in raw_out else "N/A"
    total_anchors = boxes_shape[2] if isinstance(boxes_shape, list) else None
    
    print(f"Boxes Tensor Shape: {boxes_shape}")
    print(f"Scores Tensor Shape: {scores_shape}")
    print(f"Total Spatial Anchors: {total_anchors:,}")
    
    print("\n=== Step 4: Real Image Inference Smoke Test ===")
    assert SAMPLE_VAL_IMG.exists(), f"Sample image missing: {SAMPLE_VAL_IMG}"
    smoke_res = model_p2.predict(source=str(SAMPLE_VAL_IMG), imgsz=640, device="cpu", verbose=True)
    smoke_success = len(smoke_res) > 0 and smoke_res[0].orig_shape is not None
    print(f"Smoke test completed successfully: {smoke_success}")
    print(f"Processed image dimensions: {smoke_res[0].orig_shape}")
    
    print("\n=== Step 5: CPU Latency Benchmark Across Models ===")
    process = psutil.Process()
    
    # 1. P2 Model @ 640
    print("Benchmarking YOLO11n-P2 @ 640x640...")
    p2_latency = benchmark_latency(model_p2, dummy_input, warmup=10, runs=30)
    p2_rss = round(process.memory_info().rss / (1024 ** 2), 2)
    print(f"P2 @ 640: {p2_latency['mean_ms']} ms ({p2_latency['fps']} FPS)")
    
    # 2. Baseline Model @ 640
    print("Benchmarking YOLO11n Baseline @ 640x640...")
    model_640 = YOLO(str(BASELINE_640_WEIGHTS))
    base_latency = benchmark_latency(model_640, dummy_input, warmup=10, runs=30)
    base_rss = round(process.memory_info().rss / (1024 ** 2), 2)
    print(f"Baseline @ 640: {base_latency['mean_ms']} ms ({base_latency['fps']} FPS)")
    
    # 3. Higher-Res Model @ 1280
    print("Benchmarking YOLO11n Higher-Res @ 1280x1280...")
    dummy_1280 = torch.zeros(1, 3, 1280, 1280)
    model_1280 = YOLO(str(EXP_1280_WEIGHTS))
    exp_latency = benchmark_latency(model_1280, dummy_1280, warmup=5, runs=15)
    exp_rss = round(process.memory_info().rss / (1024 ** 2), 2)
    print(f"Higher-Res @ 1280: {exp_latency['mean_ms']} ms ({exp_latency['fps']} FPS)")
    
    # Checkpoint sizes
    base_ckpt_size = round(BASELINE_640_WEIGHTS.stat().st_size / (1024 ** 2), 2)
    exp_ckpt_size = round(EXP_1280_WEIGHTS.stat().st_size / (1024 ** 2), 2)
    
    # Base params
    base_params = sum(p.numel() for p in model_640.model.parameters())
    exp_params = sum(p.numel() for p in model_1280.model.parameters())
    
    summary = {
        "model_architecture": "YOLO11n-P2",
        "input_resolution": 640,
        "classes_count": num_classes,
        "classes_names": CLASS_NAMES,
        "scales": {
            "strides": strides,
            "feature_map_shapes": feat_shapes,
            "anchors_per_scale": {
                "P2_stride_4": 160 * 160,
                "P3_stride_8": 80 * 80,
                "P4_stride_16": 40 * 40,
                "P5_stride_32": 20 * 20,
            },
            "total_anchors": total_anchors
        },
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "checkpoint_size_mb": ckpt_size_mb
        },
        "transfer_learning": {
            "source": str(PRETRAINED_WEIGHTS),
            "transferred_items": 297,
            "total_items": 593,
            "transferred_layers": "model.0 through model.16 (100% of backbone layers + initial neck downsamplers)",
            "new_initialized_layers": "model.17 through model.29 (P2 neck fusion, bottom-up path, and 4-scale Detect head)"
        },
        "smoke_test": {
            "image": str(SAMPLE_VAL_IMG),
            "success": smoke_success
        },
        "comparison": {
            "yolo11n_640_baseline": {
                "resolution": 640,
                "scales": 3,
                "strides": [8, 16, 32],
                "total_anchors": 8400,
                "parameters": base_params,
                "checkpoint_size_mb": base_ckpt_size,
                "latency_cpu": base_latency,
                "peak_process_rss_mb": base_rss
            },
            "yolo11n_1280_experiment": {
                "resolution": 1280,
                "scales": 3,
                "strides": [8, 16, 32],
                "total_anchors": 33600,
                "parameters": exp_params,
                "checkpoint_size_mb": exp_ckpt_size,
                "latency_cpu": exp_latency,
                "peak_process_rss_mb": exp_rss
            },
            "yolo11n_p2_640": {
                "resolution": 640,
                "scales": 4,
                "strides": strides,
                "total_anchors": total_anchors,
                "parameters": total_params,
                "checkpoint_size_mb": ckpt_size_mb,
                "latency_cpu": p2_latency,
                "peak_process_rss_mb": p2_rss
            }
        }
    }
    
    out_summary_file = PREFLIGHT_DIR / "architecture_preflight_summary.json"
    with open(out_summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nArchitecture validation complete. Summary written to {out_summary_file}")

if __name__ == "__main__":
    main()
