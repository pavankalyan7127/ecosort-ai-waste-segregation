# EcoSort AI - Architecture Overview

**Problem Statement 2: Smart Recycling and Waste Segregation**  
*Bharat AI-SoC Challenge 2026–27*

## 1. System Vision & Purpose
EcoSort AI is an autonomous, on-device, multilingual, and multimodal waste segregation kiosk engineered to address the critical municipal and institutional challenge of improper waste disposal at source. Designed for standalone operation on edge hardware (**Raspberry Pi 5 + AI HAT+ 2**), it eliminates dependencies on external cloud APIs, ensuring zero recurring cloud inference costs, offline operational resilience, and citizen privacy.

## 2. Hardware Specification
- **Host SBC**: Raspberry Pi 5 (Quad-core Arm Cortex-A76 @ 2.4GHz, 8GB LPDDR4X)
- **Neural Accelerator**: Raspberry Pi AI HAT+ 2 (Hailo-8 / Hailo-8L NPU providing up to 26 / 13 TOPS)
- **Vision Capture**: Raspberry Pi Camera Module 3 (Wide-angle, Sony IMX708 autofocus sensor)
- **Audio I/O**: USB Microphone Array + I2S/USB Loudspeaker (for upcoming voice interaction)
- **Display**: 10.1" / 15.6" Touchscreen Kiosk Display

## 3. Layered Software Architecture

```
+-------------------------------------------------------------+
|              React + Vite Kiosk User Interface             |
|   (Camera Feed, Classification Cards, Assistant Chat, i18n) |
+-------------------------------------------------------------+
                              |
                     REST APIs (FastAPI)
                              |
+-------------------------------------------------------------+
|                     Edge Backend Engine                     |
|                                                             |
|  +-------------------+  +--------------------------------+  |
|  |   Vision Module   |  |        Assistant Module        |  |
|  |     (YOLO11n)     |  | (Llama 3.2 Instruct ExecuTorch)|  |
|  +-------------------+  +--------------------------------+  |
|                                                             |
|  +-------------------+  +--------------------------------+  |
|  | Translation Engine|  |     Local Knowledge Base       |  |
|  |   (IndicTrans2)   |  |      (waste_rules.json)        |  |
|  +-------------------+  +--------------------------------+  |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|           Hardware Acceleration & Runtime Layers            |
|       (ExecuTorch Edge Runtime, Hailo RT, PyTorch C++)      |
+-------------------------------------------------------------+
```

## 4. Subsystem Breakdown

### 4.1 Computer Vision Subsystem (`backend/app/vision/`)
- **Model**: YOLO11n (nano) fine-tuned for recyclable and municipal waste categories.
- **Workflow**: Captures video frames, performs real-time multi-object detection, generates bounding boxes, confidence ratings, and maps detected objects to standard municipal bins.
- **Bin Categorization**:
  - **Green**: Biodegradable & Organic
  - **Blue**: Recyclables (Plastics, Paper, Metal, Glass)
  - **Red**: Domestic Hazardous & Sanitary
  - **Grey/Black**: Electronic Waste (E-Waste)

### 4.2 Conversational Assistant Subsystem (`backend/app/assistant/`)
- **Model**: Llama 3.2 Instruct (1B/3B parameters).
- **Runtime**: **ExecuTorch** (PyTorch edge inference engine) compiled with hardware-specific delegates for on-device NPU/CPU execution.
- **Function**: Explains *why* an item belongs to a specific bin, provides actionable steps (e.g., rinse milk packets, remove plastic caps), and answers civic sorting questions.

### 4.3 Multilingual & Translation Layer (`backend/app/translation/`)
- **Engine**: **IndicTrans2** (AI4Bharat).
- **Languages**: English and Tamil (`ta`) initially, expandable to 22 official Indian languages.
- **Future Audio Pipeline**:
  - **IndicConformer** for automatic speech recognition (ASR).
  - **AI4Bharat Indic-TTS** for speech generation.

### 4.4 Local Knowledge Base (`backend/app/knowledge/`)
- Deterministic, verifiable segregation rules based on national and municipal solid waste handling standards.
- Serves as the ground truth reference to anchor the assistant model's outputs.
