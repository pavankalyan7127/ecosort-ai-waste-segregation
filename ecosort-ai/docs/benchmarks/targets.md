# EcoSort AI - Target Performance Benchmarks

Performance targets for edge deployment on **Raspberry Pi 5 (8GB) + AI HAT+ 2**.

## 1. Inference Latency & Frame Rate Targets

| Task / Subsystem | Target Model | Precision / Format | Target Edge Latency | Minimum FPS |
| :--- | :--- | :--- | :--- | :--- |
| Multi-Object Detection | YOLO11n | INT8 / Hailo HEF | < 25 ms | >= 30 FPS |
| Conversational AI | Llama 3.2 Instruct (1B) | 4-bit / ExecuTorch PTE | < 80 ms / token | >= 12 tokens/sec |
| Translation (IndicTrans2) | IndicTrans2-distilled | INT8 / ONNX | < 150 ms / sentence | Interactive |
| End-to-End Segregation Loop | Combined Pipeline | Edge Optimized | < 350 ms | Responsive Kiosk |

## 2. Resource Budget

| Metric | Budget Constraint | Expected Utilization |
| :--- | :--- | :--- |
| **Peak RAM** | <= 5.5 GB (out of 8 GB) | ~3.8 GB (OS + Vision + LLM) |
| **Thermal Ceiling** | <= 75 deg C (Active Cooler) | 55 - 65 deg C continuous |
| **Storage Footprint** | <= 16 GB (on MicroSD/NVMe) | ~6.5 GB models + deps |
| **Network Dependency** | 0.0 KB (100% Offline) | Fully standalone |
