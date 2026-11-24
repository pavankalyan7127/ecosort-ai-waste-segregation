r"""Update ecosort_yolo/class_mapping.json and dataset_readme.md with refined taxonomy review.
"""

import json
from pathlib import Path

BASE_DIR = Path(r"D:\EcoSort-Resources\datasets\TACO")
ECOSORT_YOLO_DIR = BASE_DIR / "ecosort_yolo"
AUDIT_FILE = BASE_DIR / "manifests" / "final_training_readiness_audit.json"

with open(AUDIT_FILE, "r", encoding="utf-8") as f:
    audit_data = json.load(f)

classes_13 = audit_data["candidate_13_classes_detailed"]

# 1. Update class_mapping.json
mapping_path = ECOSORT_YOLO_DIR / "class_mapping.json"
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump({
        "num_classes": len(classes_13),
        "source_dataset": "TACO (Option B 640x640 Verified CC Subset, 319 Images)",
        "mapping_rationale": (
            "Maps 13 candidate Dry Packaging classes to contiguous IDs 0..12. "
            "Visually observable features are distinguished from unverified chemical resin classifications. "
            "All classes belong to statutory Dry Waste under Solid Waste Management Rules, 2026."
        ),
        "classes": classes_13
    }, f, indent=2)

print(f"Updated {mapping_path}")

# 2. Update dataset_readme.md
readme_path = ECOSORT_YOLO_DIR / "dataset_readme.md"
readme_content = f"""# EcoSort AI — YOLO Dataset Configuration (13 Candidate Classes)

**Dataset**: TACO Verified CC Subset Remapped  
**Target Architecture**: Ultralytics YOLO11n  
**Dataset Directory**: `D:\\EcoSort-Resources\\datasets\\TACO\\ecosort_yolo\\`  
**Image Dimension**: $640 \\times 640$ px (Letterbox padded with `(114, 114, 114)`)  
**Number of Classes**: 13 contiguous classes ($0..12$)  
**Total Images**: 319 (Train: 255, Val: 32, Test: 32)  
**Total Remapped Annotations**: 729 (Train: 590, Val: 73, Test: 66)

---

## 1. Class Mapping & Taxonomy Review Register

| YOLO ID | TACO ID | Class Name | EcoSort KB Mapping | SWM 2026 Stream | Mapping Status | Train | Val | Test | Total |
| :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""

for c in classes_13:
    readme_content += f"| **{c['yolo_id']}** | {c['taco_id']} | `{c['name']}` | `{c['ecosort_kb_mapping']}` | {c['statutory_stream']} | {c['mapping_status']} | {c.get('train_annotations', audit_data['all_60_classes'][c['taco_id']]['train_annotations'])} | {c.get('val_annotations', audit_data['all_60_classes'][c['taco_id']]['val_annotations'])} | {c.get('test_annotations', audit_data['all_60_classes'][c['taco_id']]['test_annotations'])} | {c.get('total_annotations', audit_data['all_60_classes'][c['taco_id']]['total_annotations'])} |\n"

readme_content += """
---

## 2. Taxonomy & Material Clarifications

1. **Plastic bottle cap (#7)**: Visually observable as rigid closure. Polymer (PP vs HDPE) cannot be resolved visually. Mapped broadly to Recyclable Rigid Plastics (Category I) under Dry Waste.
2. **Other plastic (#29)**: Generic rigid plastic debris. Resin code is unverified. Mapped broadly to non-hazardous Dry Waste for MRF sorting; not claimed specifically as HDPE.
3. **Other plastic wrapper (#39)**: Flexible plastic packaging wrapper. Single-layer vs multi-layer (MLP) cannot be established by camera. Directed to Dry Waste for MRF consolidation and co-processing/recycling triage.
4. **Styrofoam piece (#57)**: Material is Expanded Polystyrene (EPS / thermocol). The material identity is distinct from the statutory single-use ban. In municipal waste management, EPS is dry non-biodegradable waste routed to RDF/co-processing.
5. **Cigarette (#59)**: Spent cellulose acetate filter tip with ash remnant. Classified as non-recyclable combustible dry street sweepings / inerts.

---

## 3. Directory Layout

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
    ├── train\\             # 255 remapped YOLO label files (590 annotations)
    ├── val\\               # 32 remapped YOLO label files (73 annotations)
    └── test\\              # 32 remapped YOLO label files (66 annotations)
```

---

## 4. Split and Label Integrity

- Zero split leakage across 319 images.
- All normalized coordinates are strictly within $[0.0, 1.0]$.
- Original 60-class labels in `../processed/labels/` remain completely untouched.
"""

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"Updated {readme_path}")
