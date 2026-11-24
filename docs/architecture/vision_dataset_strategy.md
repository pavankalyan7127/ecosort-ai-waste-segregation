# EcoSort AI — Computer Vision Dataset Strategy & Taxonomy Mapping

**Project**: EcoSort AI (Bharat AI-SoC Challenge 2026–27)  
**Problem Statement 2**: Smart Recycling and Waste Segregation  
**Document Classification**: Technical Architecture & Dataset Engineering Specification  
**External Resource Location**: `D:\EcoSort-Resources\datasets\TACO\`  

---

## 1. Executive Summary & Problem Context

EcoSort AI requires an on-device computer vision subsystem capable of detecting multiple waste items simultaneously from camera snapshots and video streams at an autonomous waste segregation kiosk. 

To bridge computer vision with statutory reality, detections must map unambiguously to the **four statutory waste streams** established under the **Solid Waste Management Rules, 2026 (Notification S.O. 388(E))**:
1. 🟢 **Wet waste** (Kitchen, food, fruit/vegetable, flower, coconut shells)
2. 🔵 **Dry waste** (Paper, cardboard, PET/HDPE/PP plastics, metals, glass, textiles, RDF/MLP)
3. 🔴 **Sanitary waste** (Used diapers, sanitary napkins/towels, tampons, condoms, incontinence sheets)
4. ⚫ **Special care waste** (CFLs, tube lights, mercury thermometers, expired drugs, paint cans, batteries, e-waste)

No existing single off-the-shelf public dataset covers all 42 statutory knowledge base classes. Therefore, a multi-dataset aggregation, annotation normalization, and fine-tuning strategy is required.

---

## 2. Benchmark Dataset Evaluation

### 2.1 Primary Candidate: TACO (Trash Annotations in Context)
- **Official Source**: Pedro F. Proença and Pedro Simões, *TACO: Trash Annotations in Context for Litter Detection*, arXiv:2003.06975, 2020.
- **Repository / Portal**: http://tacodataset.org / https://github.com/pedropro/TACO
- **Storage Target**: `D:\EcoSort-Resources\datasets\TACO\`
- **Download Size**:
  - Full-resolution uncompressed image archive: **~3.8 GB to 4.2 GB**
  - Resized 640x640 training set: **~650 MB**
- **Annotation Format**: MS COCO JSON format (segmentation polygons and bounding boxes `[x, y, width, height]`).
- **Image Count**: ~1,500 high-resolution images containing ~5,000 annotated individual waste instances.
- **Environment Context**: Real-world litter photographed in natural settings: streets, pathways, sidewalks, parks, beaches.
- **Licensing**: Creative Commons Attribution 4.0 International (**CC BY 4.0**) — permits commercial use, adaptation, and fine-tuning with attribution.
- **Object Classes (60 subcategories across 28 supercategories)**:
  - *Plastics*: Clear plastic bottle, Other plastic bottle, Plastic bottle cap, Drink can, Food can, Plastic bag, Plastic film, Disposable plastic cup, Straw, Plastic lid, Foam cup/container.
  - *Paper*: Normal paper, Magazine paper, Corrugated cardboard, Paper bag, Paper cup, Tetra Pak carton.
  - *Glass*: Glass bottle (clear, green, brown), Broken glass shards, Glass jar.
  - *Metals*: Aluminium beverage can, Food tin can, Metal bottle cap, Scrap metal.
  - *Hazardous / Batteries*: Battery (household dry cells, small button cells).

### 2.2 Secondary Candidate: TrashBox Dataset
- **Official Source**: TrashBox Research Project (2021), Computer Vision and Pattern Recognition.
- **Repository / Download Location**: Hugging Face / GitHub TrashBox repository.
- **Expected Download Size**: **~1.2 GB**
- **Annotation Format**: Pascal VOC XML / YOLO TXT bounding boxes.
- **Image Count**: ~8,000 images across 7 high-level classes (Medical, Plastic, Glass, Metal, Paper, Cardboard, Trash).
- **Licensing**: Open academic research / MIT License.
- **Utility**: Provides supplementary medical / sanitary waste and rigid packaging instances lacking in TACO.

### 2.3 Secondary Candidate: Mendeley Bio-Medical & Sharps Dataset
- **Official Source**: Bio-Medical Waste Segregation Imagery (Mendeley Data / CPCB Studies).
- **Expected Download Size**: **~450 MB**
- **Annotation Format**: Bounding boxes for syringes, blister medicine strips, insulin needles, and yellow-bag medical waste.
- **Licensing**: CC BY-NC 4.0.
- **Utility**: Essential for training detection on **Special care waste** (insulin sharps, expired medicine strips) and **Sanitary waste** packaging.

---

## 3. Dataset-to-Taxonomy Mapping Matrix

The table below defines how TACO object annotations map to our verified 42-entry statutory knowledge base in `backend/data/waste_knowledge_base.json`:

| TACO Supercategory / Class | EcoSort Knowledge Base `class_id` | Statutory Primary Stream | Bin Assignment | Mapping Reliability |
| :--- | :--- | :--- | :--- | :--- |
| `Bottle > Clear plastic bottle` | `dry_plastic_pet_bottles` | Dry waste | Blue Bin | High (Direct 1:1) |
| `Bottle > Other plastic bottle` | `dry_plastic_hdpe_containers` | Dry waste | Blue Bin | High (Direct 1:1) |
| `Bottle cap > Plastic cap` | `dry_plastic_pp_containers` | Dry waste | Blue Bin | High (Direct 1:1) |
| `Can > Drink can` | `dry_metal_aluminium_cans` | Dry waste | Blue Bin | High (Direct 1:1) |
| `Can > Food can` | `dry_metal_steel_tin_cans` | Dry waste | Blue Bin | High (Direct 1:1) |
| `Carton > Drink carton` | `dry_paper_tetra_packs` | Dry waste | Blue Bin | High (Direct 1:1) |
| `Carton > Corrugated carton` | `dry_paper_cardboard_cartons` | Dry waste | Blue Bin | High (Direct 1:1) |
| `Paper > Normal paper` | `dry_paper_newspaper` | Dry waste | Blue Bin | High (Direct 1:1) |
| `Glass bottle > Clear / Green / Brown` | `dry_glass_bottles_jars` | Dry waste | Blue Bin | High (Direct 1:1) |
| `Broken glass` | `dry_glass_broken_shards` | Dry waste | Blue Bin (Wrap first) | High (Direct 1:1) |
| `Plastic film > Crisp packet` | `dry_plastic_mlp_non_recyclable` | Dry waste (RDF) | Blue Bin | High (Direct 1:1) |
| `Plastic film > Wrapper` | `dry_plastic_mlp_non_recyclable` | Dry waste (RDF) | Blue Bin | High (Direct 1:1) |
| `Plastic film > Plastic bag` | `dry_plastic_single_use_banned` | Dry waste (RDF) | Blue Bin | High (Direct 1:1) |
| `Cup > Disposable plastic cup` | `dry_plastic_single_use_banned` | Dry waste (RDF) | Blue Bin | High (Direct 1:1) |
| `Straw > Plastic straw` | `dry_plastic_single_use_banned` | Dry waste (RDF) | Blue Bin | High (Direct 1:1) |
| `Battery` | `special_batteries_household_dry_cell` | Special care waste | Deposition Centre | Moderate (Small subset) |

---

## 4. Annotation Gaps & Mitigation Strategy

Our audit reveals that existing open-access litter datasets exhibit critical gaps when evaluated against India's **Solid Waste Management Rules, 2026**:

### Gap 1: Organic / Wet Waste Under-Representation
- **Deficiency**: TACO and street litter datasets prioritize persistent packaging litter. Organic items (cooked food scraps, vegetable peels, fish/meat scraps, coconut shells) rapidly decompose and are rarely photographed in public street datasets.
- **Mitigation**: Supplement with food/agricultural object datasets (e.g. Food-101, Fruits-360) and acquire ~500 customized high-angle kiosk images capturing Indian kitchen waste: banana peels, onion skins, coconut shells, and tea grounds.

### Gap 2: Sanitary Waste Absence
- **Deficiency**: Standard open datasets have zero or negligible annotations for used diapers, sanitary pads, or menstrual hygiene pouches due to privacy, hygiene taboos, and ethical collection barriers.
- **Mitigation**: Train the vision pipeline on **packaged/wrapped sanitary waste pouches** (Rule 5(1)(c) mandates that citizens wrap used sanitary items in disposal pouches before binning). Synthetic mock training using wrapped disposal pouches marked with the statutory red dot indicator will be compiled.

### Gap 3: Specialized Domestic Hazardous Waste
- **Deficiency**: CFL bulbs, mercury thermometers, and paint drums appear in very low frequencies in generic litter datasets.
- **Mitigation**: Curate a dedicated ~300 image subset for CFLs, tube lights, and pesticide cans from industrial hazardous imagery and laboratory collections.

---

## 5. Storage & Fine-Tuning Execution Plan

In accordance with storage isolation rules:
- **Dataset Storage**: All downloaded raw images and YOLO-formatted label text files will reside exclusively in:
  `D:\EcoSort-Resources\datasets\TACO\`
- **Pretrained Weights**: `D:\EcoSort-Resources\models\pretrained\`
- **Fine-Tuned Checkpoints**: `D:\EcoSort-Resources\models\fine-tuned\`
- **Training Runs & Logs**: `D:\EcoSort-Resources\experiments\`

### Download Notification Protocol
Before executing bulk downloads, the system reports:
- Dataset name and version
- Exact download URL
- Size on disk (uncompressed and compressed)
- Target drive free space verification
and awaits explicit operator confirmation.
