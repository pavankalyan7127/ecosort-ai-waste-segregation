"""Comprehensive test suite for the EcoSort AI computer vision pipeline."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.vision.class_mapping import WasteClassMapper
from app.vision.detector import WasteDetector
from app.vision.model_manager import ModelManager
from app.vision.postprocessing import clamp, filter_and_format_detections, normalize_bounding_box
from app.vision.preprocessing import ImageValidationError, validate_and_load_image


def create_test_image(width: int = 640, height: int = 480, color: str = "red", format: str = "JPEG") -> bytes:
    """Helper to generate in-memory test image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def create_rgba_test_image(width: int = 100, height: int = 100) -> bytes:
    """Helper to generate RGBA image with alpha channel."""
    img = Image.new("RGBA", (width, height), color=(0, 255, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------
# 1. Image Preprocessing Tests
# ----------------------------------------------------------------------

def test_validate_and_load_valid_image():
    image_bytes = create_test_image(640, 480)
    img, (w, h, c) = validate_and_load_image(image_bytes)
    assert w == 640
    assert h == 480
    assert c == 3
    assert img.mode == "RGB"


def test_validate_and_load_rgba_converts_to_rgb():
    image_bytes = create_rgba_test_image(128, 128)
    img, (w, h, c) = validate_and_load_image(image_bytes)
    assert w == 128
    assert h == 128
    assert c == 3
    assert img.mode == "RGB"


def test_validate_and_load_empty_payload():
    with pytest.raises(ImageValidationError, match="Empty image payload"):
        validate_and_load_image(b"")


def test_validate_and_load_corrupted_payload():
    with pytest.raises(ImageValidationError, match="Invalid or corrupted"):
        validate_and_load_image(b"not_an_image_random_bytes_12345")


def test_validate_and_load_too_small_dimensions():
    tiny_image = create_test_image(16, 16)
    with pytest.raises(ImageValidationError, match="too small"):
        validate_and_load_image(tiny_image)


# ----------------------------------------------------------------------
# 2. Post-processing & Bounding Box Normalization Tests
# ----------------------------------------------------------------------

def test_clamp_function():
    assert clamp(-0.5) == 0.0
    assert clamp(1.5) == 1.0
    assert clamp(0.45) == 0.45


def test_normalize_bounding_box_valid():
    box = (100.0, 120.0, 300.0, 360.0)
    norm = normalize_bounding_box(box, image_width=1000, image_height=1000)
    assert norm["x_min"] == 0.1
    assert norm["y_min"] == 0.12
    assert norm["x_max"] == 0.3
    assert norm["y_max"] == 0.36


def test_normalize_bounding_box_out_of_bounds_clamping():
    box = (-50.0, -20.0, 1100.0, 1200.0)
    norm = normalize_bounding_box(box, image_width=1000, image_height=1000)
    assert norm["x_min"] == 0.0
    assert norm["y_min"] == 0.0
    assert norm["x_max"] == 1.0
    assert norm["y_max"] == 1.0


def test_normalize_bounding_box_invalid_dimensions():
    with pytest.raises(ValueError, match="Invalid image dimensions"):
        normalize_bounding_box((10, 10, 50, 50), image_width=0, image_height=100)


def test_filter_and_format_detections():
    raw_detections = [
        {"label": "bottle", "confidence": 0.88, "box_pixels": (10, 10, 50, 50)},
        {"label": "cup", "confidence": 0.20, "box_pixels": (20, 20, 60, 60)},  # below threshold
        {"label": "banana", "confidence": 0.95, "box_pixels": (30, 30, 70, 70)},
    ]
    filtered = filter_and_format_detections(
        raw_detections=raw_detections,
        image_width=100,
        image_height=100,
        confidence_threshold=0.25,
    )
    assert len(filtered) == 2
    # banana (0.95) should be first, then bottle (0.88)
    assert filtered[0]["predicted_label"] == "banana"
    assert filtered[0]["confidence"] == 0.95
    assert filtered[1]["predicted_label"] == "bottle"
    assert filtered[1]["confidence"] == 0.88


# ----------------------------------------------------------------------
# 3. Class Mapping & Statutory Knowledge Integration Tests
# ----------------------------------------------------------------------

def test_class_mapping_supported_coco_classes():
    mapper = WasteClassMapper()

    # Dry waste check (bottle -> dry_plastic_pet_bottles)
    bottle_res = mapper.map_detection("bottle")
    assert bottle_res["is_waste_mapped"] is True
    assert bottle_res["primary_stream"] == "Dry waste"
    assert bottle_res["mapped_class_id"] == "dry_plastic_pet_bottles"
    assert "PET" in bottle_res["item_name"]
    assert len(bottle_res["source_references"]) > 0

    # Wet waste check (banana -> wet_fruit_vegetable_peels)
    banana_res = mapper.map_detection("banana")
    assert banana_res["is_waste_mapped"] is True
    assert banana_res["primary_stream"] == "Wet waste"
    assert banana_res["bin_color"] == "Green"

    # Special care waste check (cell phone -> special_ewaste_smartphones_tablets)
    phone_res = mapper.map_detection("cell phone")
    assert phone_res["is_waste_mapped"] is True
    assert phone_res["primary_stream"] == "Special care waste"
    assert phone_res["mapped_class_id"] == "special_ewaste_smartphones_tablets"


def test_class_mapping_unsupported_classes_not_hallucinated():
    mapper = WasteClassMapper()
    # Unrelated COCO classes MUST NOT be mapped to waste
    for unrelated in ["person", "car", "dog", "chair", "airplane"]:
        res = mapper.map_detection(unrelated)
        assert res["is_waste_mapped"] is False
        assert res["primary_stream"] == "Unsupported"
        assert res["mapped_class_id"] is None


def test_class_mapping_waste_specific_taco_classes():
    mapper = WasteClassMapper()
    res = mapper.map_detection("drink can", model_type="waste_specific_finetuned")
    assert res["is_waste_mapped"] is True
    assert res["primary_stream"] == "Dry waste"
    assert res["mapped_class_id"] == "dry_metal_aluminium_cans"


# ----------------------------------------------------------------------
# 4. Model Manager Tests
# ----------------------------------------------------------------------

def test_model_manager_missing_weights_graceful_handling():
    manager = ModelManager()
    info = manager.get_model("non_existent_weights_xyz.pt")
    assert info["is_loaded"] is False
    assert info["status"] == "model_weights_not_found"
    assert "Download confirmation required" in info["error_message"]


# ----------------------------------------------------------------------
# 5. WasteDetector Pipeline Tests
# ----------------------------------------------------------------------

def test_waste_detector_mock_detections_pipeline():
    detector = WasteDetector()
    image_bytes = create_test_image(640, 480)

    mock_dets = [
        {"label": "bottle", "confidence": 0.92, "box_pixels": (100, 100, 200, 300)},
        {"label": "banana", "confidence": 0.85, "box_pixels": (300, 200, 450, 350)},
        {"label": "person", "confidence": 0.90, "box_pixels": (50, 50, 400, 400)},  # unmapped
        {"label": "cup", "confidence": 0.15, "box_pixels": (10, 10, 40, 40)},       # low conf
    ]

    result = detector.detect(
        image_bytes=image_bytes,
        confidence_threshold=0.25,
        _mock_detections=mock_dets,
    )

    assert result["status"] == "success"
    assert result["total_detected"] == 3  # cup excluded (< 0.25)
    assert result["total_waste_mapped"] == 2  # bottle + banana (person is unmapped)
    assert result["stream_counts"]["Dry waste"] == 1
    assert result["stream_counts"]["Wet waste"] == 1

    # Check bounding box normalization of bottle
    bottle_det = next(d for d in result["detected_items"] if d["detected_label"] == "bottle")
    assert bottle_det["is_waste_mapped"] is True
    assert bottle_det["primary_stream"] == "Dry waste"
    assert 0.0 <= bottle_det["bounding_box"]["x_min"] <= 1.0

    # Check unmapped person
    person_det = next(d for d in result["detected_items"] if d["detected_label"] == "person")
    assert person_det["is_waste_mapped"] is False
    assert person_det["primary_stream"] == "Unsupported"


def test_waste_detector_without_weights():
    detector = WasteDetector()
    image_bytes = create_test_image(640, 480)

    # When no model weights exist on disk, returns model_weights_not_found
    result = detector.detect(image_bytes=image_bytes, model_name="unloaded_model_123.pt")
    assert result["status"] == "model_weights_not_found"
    assert result["pipeline_status"] == "weights_pending_download"
    assert result["total_detected"] == 0
    assert result["detected_items"] == []


# ----------------------------------------------------------------------
# 6. FastAPI Endpoint Tests (/api/vision/detect)
# ----------------------------------------------------------------------

client = TestClient(app)


def test_api_vision_detect_valid_image():
    image_bytes = create_test_image(640, 480)
    files = {"file": ("test_kiosk_frame.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    response = client.post("/api/vision/detect", files=files, data={"confidence_threshold": "0.3"})

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "target_hardware" in data
    assert data["target_hardware"] == "Raspberry Pi 5 + AI HAT+ 2 (Hailo-8 / Hailo-8L NPU)"
    assert "detected_items" in data
    assert isinstance(data["detected_items"], list)


def test_api_vision_detect_invalid_corrupted_image():
    files = {"file": ("corrupt.jpg", io.BytesIO(b"invalid_binary_stream_garbage"), "image/jpeg")}
    response = client.post("/api/vision/detect", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "Invalid or corrupted image" in data["detail"]
