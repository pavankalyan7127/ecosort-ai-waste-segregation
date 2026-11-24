# EcoSort AI 🌿

> **AI-Powered, Multilingual, Multimodal, Standalone Waste Segregation Kiosk**  
> **Bharat AI-SoC Challenge 2026–27** &bull; **Problem Statement 2: Smart Recycling and Waste Segregation**

---

## 📌 Project Overview

**EcoSort AI** is an edge-first, completely standalone intelligent kiosk engineered for high-throughput public, campus, and municipal waste sorting. By pairing computer vision with edge-optimized conversational intelligence and multilingual localization, EcoSort AI guides citizens to segregate waste accurately into proper bins right at the point of disposal.

### Core Capabilities
1. **Live Camera & Photo Ingestion**: Processes real-time camera frames or user-uploaded photographs.
2. **Multi-Object Detection**: Detects and localizes multiple waste items simultaneously using an edge-optimized **YOLO11n** model.
3. **Automated Category Mapping**: Classifies detected items into standard color-coded categories:
   - 🟢 **Green Bin**: Wet / Biodegradable / Organic Waste
   - 🔵 **Blue Bin**: Dry / Recyclable Waste (Paper, Cardboard, Plastics, Glass, Metals)
   - 🔴 **Red Bin**: Domestic Hazardous & Sanitary Waste
   - ⚫ **Grey / Black Bin**: Electronic Waste (E-Waste)
4. **Deterministic Local Knowledge Base**: Backed by a verified municipal waste disposal ruleset (`waste_rules.json`).
5. **On-Device Conversational AI**: Integrates **Llama 3.2 Instruct** running locally via PyTorch's **ExecuTorch** runtime for real-time guidance without cloud APIs.
6. **Bilingual & Indian Language Support**: Natively supports **English** and **Tamil** via **IndicTrans2**, designed to scale to all 22 official Indian languages.
7. **Edge Hardware Acceleration**: Specifically architected for the **Raspberry Pi 5 + AI HAT+ 2 (Hailo-8)** platform.
8. **Future Multimodal Speech**: Prepared for voice input via **IndicConformer** (ASR) and voice output via **AI4Bharat Indic-TTS**.

---

## 🏗️ System Architecture

```
ecosort-ai/
├── frontend/                     # Kiosk Web UI (Vite + React 18 + React Router + Tailwind CSS + Axios)
│   ├── public/                   # Static assets & icons
│   ├── src/
│   │   ├── api/
│   │   │   └── client.js         # Axios client for backend REST communication
│   │   ├── components/
│   │   │   ├── Header.jsx        # Navigation, hardware status & language switcher
│   │   │   ├── CameraView.jsx    # Live camera feed, uploads & bounding box overlay
│   │   │   ├── DetectionCard.jsx # Waste category & bin assignment directives
│   │   │   ├── AssistantChat.jsx # Llama 3.2 ExecuTorch conversational interface
│   │   │   └── CategoryLegend.jsx# Color-coded segregation guidelines
│   │   ├── pages/
│   │   │   ├── KioskHome.jsx     # Main interactive kiosk feed and detection view
│   │   │   └── WasteGuide.jsx    # Full categorical search and segregation directory
│   │   ├── App.jsx               # React Router DOM navigation container
│   │   ├── index.css             # Tailwind base & kiosk styling
│   │   └── main.jsx              # React DOM entrypoint (pure JavaScript)
│   ├── index.html                # Kiosk single-page shell
│   ├── package.json              # Node dependencies & npm scripts
│   ├── tailwind.config.js        # Tailwind CSS configuration
│   └── vite.config.js            # Vite build & dev server config
│
├── backend/                      # Edge Inference & REST API Backend (FastAPI + Uvicorn)
│   ├── .venv/                    # Python 3.12 virtual environment
│   ├── app/
│   │   ├── api/
│   │   │   └── router.py         # REST endpoints for vision, assistant, knowledge & i18n
│   │   ├── assistant/
│   │   │   └── executorch_llm.py # Llama 3.2 Instruct ExecuTorch runtime wrapper
│   │   ├── core/
│   │   │   └── config.py         # Central system configuration & path settings
│   │   ├── knowledge/
│   │   │   └── waste_db.py       # Rule engine reading local waste taxonomy
│   │   ├── translation/
│   │   │   └── translator.py     # IndicTrans2 multilingual translation layer
│   │   ├── vision/
│   │   │   ├── categories.py     # Pydantic schemas & waste category types
│   │   │   └── detector.py       # YOLO11n object detector edge interface
│   │   └── main.py               # FastAPI application entrypoint with health-check & CORS
│   ├── data/
│   │   └── waste_rules.json      # Structured local waste knowledge base
│   ├── models/                   # Directory for offline model weights (*.pt, *.pte, *.hef)
│   │   └── README.md             # Model weight acquisition and format guidelines
│   ├── .env.example              # Sample environment configuration
│   └── requirements.txt          # Python backend dependencies
│
├── docs/
│   ├── architecture/
│   │   └── overview.md           # Detailed technical design & edge pipeline flow
│   └── benchmarks/
│       └── targets.md            # Hardware latency, FPS & memory constraints
├── .gitignore                    # Excludes venvs, model weights, node_modules & secrets
└── README.md                     # Project documentation
```

---

## 🛠️ Technology Stack & Dependencies

### Frontend Dependencies (`package.json`)
- **React (`^18.3.1`)** & **React DOM (`^18.3.1`)**: Component-based UI framework (**JavaScript only**, no TypeScript).
- **Vite (`^5.4.11`)**: Fast dev server and asset bundler.
- **React Router DOM (`^7.18.5`)**: Client-side routing between Kiosk Mode and the Segregation Guide.
- **Tailwind CSS (`^3.4.17`)**: Utility-first styling for dark-themed touchscreen kiosk interfaces.
- **Lucide React (`^0.460.0`)**: Modern iconography.
- **Axios (`^1.21.0`)**: Promise-based HTTP client for FastAPI REST endpoints.

### Backend Dependencies (`requirements.txt`)
- **FastAPI (`>=0.115.0`)**: High-performance asynchronous web framework for REST APIs.
- **Uvicorn (`>=0.30.0`)**: ASGI web server.
- **Pydantic (`>=2.7.0`)**: Data validation and response schemas.
- **Python-multipart (`>=0.0.9`)**: Handling multipart form data for camera and image uploads.
- **Pillow (`>=10.0.0`)**: Image processing and format manipulation.
- **NumPy (`>=1.26.0`)**: Numerical array operations.
- **OpenCV (`opencv-python-headless>=4.10.0`)**: Computer vision utilities and frame decoding.
- **Ultralytics (`>=8.3.0`)**: YOLO11n object detection engine.
- **Requests (`>=2.31.0`)**: HTTP client utilities.
- **Python-dotenv (`>=1.0.0`)**: Loading configuration from `.env` files.

> **Note on PyTorch & ExecuTorch Separation:**  
> PyTorch is automatically utilized as the backend runtime for Ultralytics during development. ExecuTorch and Hailo NPU runtimes are kept isolated for target SoC deployment on the Raspberry Pi 5 + AI HAT+ 2 hardware to prevent unnecessary or incompatible Windows-specific binary overhead.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- **Node.js**: v18+ (verified on v22)
- **Python**: 3.10+ (verified on 3.12)
- **Git**

### 2. Backend Setup

```powershell
# Navigate to backend directory
cd ecosort-ai\backend

# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\.venv\Scripts\activate.bat
# Linux / Raspberry Pi OS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```powershell
# Navigate to frontend directory
cd ecosort-ai\frontend

# Install node dependencies
npm install
```

---

## 🖥️ Starting the Development Servers on Windows

Open two separate PowerShell terminal windows:

### Terminal 1 — Backend (FastAPI)
```powershell
cd "c:\Users\pavan\Documents\Bharat AI-SoC Challenge — 2026–27\ecosort-ai\backend"
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Direct Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **API Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Terminal 2 — Frontend (Vite + React)
```powershell
cd "c:\Users\pavan\Documents\Bharat AI-SoC Challenge — 2026–27\ecosort-ai\frontend"
npm run dev
```
- **Kiosk Web Interface**: [http://localhost:5173](http://localhost:5173)

---

## 🧪 Health-Check Verification

You can verify the backend health endpoint using PowerShell:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

**Expected JSON Output:**
```json
{
  "status": "healthy",
  "app": "EcoSort AI",
  "version": "0.1.0",
  "target_hardware": "Raspberry Pi 5 + AI HAT+ 2",
  "supported_languages": [
    "en",
    "ta"
  ]
}
```
