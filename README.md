# EcoSort AI 🌿
### AI-Powered, Multilingual, Multimodal Waste Segregation Kiosk
**Bharat AI-SoC Challenge 2026–27** &bull; **Problem Statement 2: Smart Recycling & Waste Segregation**  
*Target Hardware Platform: Raspberry Pi 5 (8 GB) + AI HAT+ 2 (Hailo-8 / Hailo-8L NPU, 13–26 TOPS)*

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg)](https://vitejs.dev/)
[![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-48%2F48%20Passed-brightgreen.svg)]()

---

## 📌 Executive Summary

**EcoSort AI** is an edge-first, completely standalone intelligent kiosk engineered for high-throughput public, campus, and municipal waste sorting. By coupling on-device computer vision with deterministic statutory classification and bilingual conversational guidance, EcoSort AI directs citizens to segregate waste accurately into proper bins right at the point of disposal.

Unlike traditional cloud-dependent smart bins, EcoSort AI is designed to run **100% offline** on edge silicon within a strict **<15W power envelope**, eliminating recurring cloud inference costs, latency bottlenecks, and citizen privacy concerns.

```
                            SOLID WASTE GENERATION AT SOURCE
                                         │
       ┌───────────────────┬─────────────┴─────────────┬───────────────────┐
       ▼                   ▼                           ▼                   ▼
 🟢 GREEN BIN         🔵 BLUE BIN                 🔴 RED BIN          ⚫ BLACK / GREY BIN
  Wet / Organic       Dry / Recyclable            Sanitary Waste      Special Care & E-Waste
  (Compost/Biogas)    (PET, HDPE, Cans, Paper)    (Diapers, Napkins)  (Batteries, E-Waste, CFLs)
```

---

## 🏛️ Statutory Regulatory Grounding (Zero Hallucination)

EcoSort AI eliminates artificial hallucination by anchoring every disposal decision, handling precaution, and preparation step directly in authentic statutory legislation:

1. **Solid Waste Management Rules, 2026** (*Notification S.O. 388(E)*, effective April 2026) — Mandatory 4-stream segregation protocol.
2. **E-Waste (Management) Rules, 2022** (*Notification G.S.R. 814(E)*).
3. **Battery Waste Management Rules, 2022** (*Notification G.S.R. 664(E)*).
4. **Plastic Waste Management Rules, 2016** (as amended through 2024, single-use plastic ban).
5. **Government of Tamil Nadu G.O. (Ms.) No. 84** (Environment and Forests Dept, 14 single-use plastic item ban).

---

## 🚀 Key Features

* **🎥 Live Camera & Video Ingestion**: Real-time webcam streaming via WebRTC (`navigator.mediaDevices.getUserMedia`) with interactive snapshotting and continuous 3-second auto-scan kiosk mode.
* **📦 Multi-Object Detection**: Simultaneous multi-object detection and bounding box localization powered by fine-tuned **YOLO11n** running at **22+ FPS** on CPU (projected at **>100 FPS** on the Hailo-8 NPU).
* **🎨 Color-Coded Dynamic Overlays**: Detected items are enclosed in bounding boxes color-coded to their destination bin (🟢 Green, 🔵 Blue, 🔴 Red, ⚫ Amber/Black).
* **🌐 Bilingual Localization**: Natively supports **English** and **Tamil (`ta`)**, expandable to all 22 scheduled Indian languages via AI4Bharat IndicTrans2.
* **📖 Deterministic 42-Item Waste Knowledge Base**: Comprehensive repository with preparation guidance (e.g., rinse milk packets, remove caps, wrap sanitary waste).
* **💬 Conversational Kiosk Assistant**: On-device natural language interface powered by **Llama 3.2 Instruct (1B)** via PyTorch **ExecuTorch**.
* **⚡ Target Edge Silicon**: Optimized for the **Raspberry Pi 5 + AI HAT+ 2 (Hailo-8 / Hailo-8L NPU)**.

---

## 🏗️ Architecture & Technology Stack

```
ecosort-ai/
├── frontend/                     # Kiosk Touchscreen Web UI (React 18 + Vite 5 + Tailwind CSS)
│   ├── src/
│   │   ├── api/client.js         # Axios client communicating with FastAPI
│   │   ├── components/
│   │   │   ├── CameraView.jsx    # Live webcam feed, video capture & dynamic bounding box overlay
│   │   │   ├── DetectionCard.jsx # Waste directives, confidence ratings & bin assignment cards
│   │   │   ├── AssistantChat.jsx # Conversational assistant interface
│   │   │   ├── Header.jsx        # Navigation, hardware status & language switcher (EN / தமிழ்)
│   │   │   └── CategoryLegend.jsx# Color-coded segregation guide
│   │   └── pages/
│   │       ├── KioskHome.jsx     # Main interactive kiosk feed and detection view
│   │       └── WasteGuide.jsx    # Searchable 42-item statutory waste directory
│
├── backend/                      # Edge Inference & REST API Backend (FastAPI + Uvicorn)
│   ├── app/
│   │   ├── api/router.py         # REST endpoints for vision, knowledge base, assistant & i18n
│   │   ├── vision/
│   │   │   ├── detector.py       # YOLO11n object detector & coordinate normalization
│   │   │   ├── model_manager.py  # Singleton weight loader (waste-specific & COCO fallback)
│   │   │   ├── class_mapping.py  # Bridges model labels to statutory 4-stream taxonomy
│   │   │   ├── preprocessing.py  # Image byte validation, dimension clamping, RGB conversion
│   │   │   └── postprocessing.py # Boundary clamping [0,1] & confidence threshold filtering
│   │   ├── knowledge/            # Local deterministic rule engine reading waste_rules.json
│   │   └── core/config.py        # Central configuration & hardware path resolution
│   ├── data/waste_rules.json     # 42 statutory waste items grounded in SWM Rules 2026
│   └── tests/                    # 48 automated pytest regression unit and integration tests
│
└── docs/                         # Technical architecture specifications & training benchmark reports
    ├── architecture/             # System design, CV pipeline, and knowledge base specs
    └── benchmarks/               # Empirical benchmark logs, confusion matrices, and audit reports
```

---

## 📊 Computer Vision Benchmarks & Experiments

Across multiple experimental iterations, we evaluated standard 3-scale YOLO11n and custom 4-scale YOLO11n-P2 architectures on an audited 319-image commercial-permissive subset of the **TACO (Trash Annotations in Context)** dataset:

| Dimension / Metric | 1. Baseline 640 | 2. High-Res 1280 | 3. P2 Baseline | 4. P2 Remediated V1 |
| :--- | :---: | :---: | :---: | :---: |
| **Input Resolution** | $640 \times 640$ | $1280 \times 1280$ | $640 \times 640$ | **$640 \times 640$** |
| **Detection Scales** | 3 ($P3, P4, P5$) | 3 ($P3, P4, P5$) | 4 ($P2\text{--}P5$) | **4 ($P2\text{--}P5$)** |
| **Spatial Anchors** | $8,400$ | $33,600$ | $34,000$ | **$34,000$** |
| **Parameters** | **2.61 M** | 2.61 M | 2.67 M | **2.67 M** |
| **CPU Latency (mean)** | **$45.10\text{ ms}$** | $194.10\text{ ms}$ | $77.40\text{ ms}$ | **$77.10\text{ ms}$** |
| **Inference Throughput** | **$22.2\text{ FPS}$** | $5.15\text{ FPS}$ | $12.9\text{ FPS}$ | **$13.0\text{ FPS}$** |
| **Completed Epochs** | 50 (Full) | 47 (Early stop) | 20 (Early stop) | **35 (Full schedule)** |
| **Max Confidence** | $92.40\%$ | $93.10\%$ | $15.41\%$ | **$70.74\%$** |
| **Macro Precision** | $32.07\%$ | **$42.96\%$** | $0.48\%$ | **$31.04\%$** |
| **Macro mAP@50** | $14.71\%$ | **$19.13\%$** | $5.57\%$ | **$8.74\%$** |

> Complete benchmark curves, diagnostic visualizations, and confusion matrices are preserved in [`docs/benchmarks/`](docs/benchmarks/).

---

## 🛠️ Quickstart: Running Locally

### 1. Prerequisites
- **Python**: 3.10+ (tested on Python 3.12)
- **Node.js**: v18+ (tested on Node v22)
- **Git**

### 2. Backend Setup
```bash
# Navigate to the backend directory
cd ecosort-ai/backend

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*API Swagger Documentation will be available at: `http://localhost:8000/docs`*

### 3. Frontend Setup
In a new terminal:
```bash
# Navigate to the frontend directory
cd ecosort-ai/frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
*Open your browser and navigate to: **`http://localhost:5173`***

---

## 🧪 Running Automated Tests

The backend includes a comprehensive 48-test regression suite validating API schemas, image decoding, boundary clamping, and zero-hallucination statutory mapping:

```bash
cd ecosort-ai/backend
pytest -v
```

---

## 📜 Intellectual Property & Dataset Licensing

* **Software License**: Distributed under the MIT License.
* **Statutory Knowledge**: Grounded in official public Gazette notifications issued by the Ministry of Environment, Forest and Climate Change (MoEFCC), Government of India.
* **Dataset Attribution**: Trained using an independently audited, Creative Commons-compliant ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) & [CC0](https://creativecommons.org/publicdomain/zero/1.0/)) subset of the **TACO (Trash Annotations in Context)** dataset (*Pedro F. Proença and Pedro Simões, 2020*). Non-commercial Flickr contributions were audited and excluded to ensure unrestricted deployment compliance.

---

## 👥 Authors & Acknowledgments

* **Project**: EcoSort AI
* **Competition**: Bharat AI-SoC Challenge 2026–27
* **Problem Statement**: PS-02: Smart Recycling & Waste Segregation
