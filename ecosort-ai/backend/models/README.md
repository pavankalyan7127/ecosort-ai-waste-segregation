# EcoSort AI - Model Storage Directory

This directory stores offline AI model weights and artifacts for on-device inference on the Raspberry Pi 5 + AI HAT+ 2 platform.

## Target Models

1. **Object Detection**:
   - Model: `YOLO11n` (nano)
   - Format: `yolo11n.pt` / `yolo11n.onnx` / Hailo HEF (compiled for Raspberry Pi AI HAT+ 2)
   - Function: Real-time multi-object waste detection from camera streams.

2. **Conversational Assistant**:
   - Model: `Llama 3.2 Instruct` (1B / 3B)
   - Format: `llama3.2_instruct.pte` (ExecuTorch package optimized for edge SoC acceleration)
   - Function: Natural language disposal guidance and citizen Q&A.

3. **Machine Translation**:
   - Model: `IndicTrans2` (1B / 200M distilled)
   - Target languages: English (`en`) <-> Tamil (`ta`) initially, expandable to 22 scheduled Indian languages.

4. **Speech Components (Future Integration)**:
   - ASR: `IndicConformer`
   - TTS: `AI4Bharat Indic-TTS`

*Note: Large model weights (`*.pt`, `*.pte`, `*.onnx`, `*.bin`, `*.hef`) are excluded from version control via `.gitignore`.*
