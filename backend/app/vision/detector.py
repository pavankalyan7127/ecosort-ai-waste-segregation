"""Waste object detector integrating YOLO inference, image preprocessing,

coordinate normalization, and statutory knowledge base mapping.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from PIL import Image

from .class_mapping import WasteClassMapper
from .model_manager import ModelManager
from .postprocessing import filter_and_format_detections
from .preprocessing import ImageValidationError, validate_and_load_image

logger = logging.getLogger(__name__)


class WasteDetector:
    """High-level detector orchestrating pre-processing, YOLO inference,

    post-processing coordinate normalization, and statutory waste mapping.
    """

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        class_mapper: Optional[WasteClassMapper] = None,
    ):
        self.model_manager = model_manager or ModelManager()
        self.class_mapper = class_mapper or WasteClassMapper()

    def detect(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.25,
        model_name: Optional[str] = None,
        _mock_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Detect waste objects in an image byte payload.

        Args:
            image_bytes: Raw binary image data.
            confidence_threshold: Minimum confidence score [0.0, 1.0].
            model_name: Optional model weights filename (defaults to yolo11n.pt).
            _mock_detections: Optional raw detections for deterministic unit testing.

        Returns:
            Dict structured according to VisionDetectResponse schema.
        """
        # Step 1: Validate and decode image stream safely
        pil_image, (width, height, channels) = validate_and_load_image(image_bytes)

        # Step 2: Check for test mock detections first
        if _mock_detections is not None:
            return self._process_detection_results(
                raw_detections=_mock_detections,
                image_width=width,
                image_height=height,
                confidence_threshold=confidence_threshold,
                model_type="coco_demonstration",
                model_name="mock_test_model",
                latency_ms=1.5,
            )

        # Step 3: Check model manager for active model weights
        model_info = self.model_manager.get_model(model_name)

        if not model_info["is_loaded"]:
            # Model weights not found on disk; return transparent, un-fabricated status
            logger.info("Model weights not present on disk. Returning pending status.")
            return {
                "status": "model_weights_not_found",
                "message": (
                    f"Pipeline is operational, but model weights '{model_info['model_name']}' "
                    f"were not found in external storage ({model_info['model_path']}). "
                    "Confirmation is required before downloading weights."
                ),
                "target_hardware": "Raspberry Pi 5 + AI HAT+ 2 (Hailo-8 / Hailo-8L NPU)",
                "target_model": model_info["model_name"],
                "pipeline_status": "weights_pending_download",
                "is_integrated": True,
                "model_type": model_info["model_type"],
                "image_dimensions": {"width": width, "height": height, "channels": channels},
                "inference_latency_ms": 0.0,
                "total_detected": 0,
                "total_waste_mapped": 0,
                "stream_counts": {},
                "detected_items": [],
            }

        # Step 4: Perform real inference
        yolo_model = model_info["model"]
        start_time = time.perf_counter()
        results = yolo_model(pil_image, conf=confidence_threshold, verbose=False)
        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)

        raw_detections: List[Dict[str, Any]] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names or {}
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label_name = names.get(cls_id, f"class_{cls_id}")
                raw_detections.append({
                    "label": label_name,
                    "confidence": conf,
                    "box_pixels": tuple(xyxy),
                })

        return self._process_detection_results(
            raw_detections=raw_detections,
            image_width=width,
            image_height=height,
            confidence_threshold=confidence_threshold,
            model_type=model_info["model_type"],
            model_name=model_info["model_name"],
            latency_ms=latency_ms,
        )

    def _process_detection_results(
        self,
        raw_detections: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        confidence_threshold: float,
        model_type: str,
        model_name: str,
        latency_ms: float,
    ) -> Dict[str, Any]:
        """Normalize coordinates, filter by threshold, map to statutory taxonomy,

        and aggregate stream counts.
        """
        filtered = filter_and_format_detections(
            raw_detections=raw_detections,
            image_width=image_width,
            image_height=image_height,
            confidence_threshold=confidence_threshold,
        )

        detected_items = []
        stream_counts: Dict[str, int] = {
            "Wet waste": 0,
            "Dry waste": 0,
            "Sanitary waste": 0,
            "Special care waste": 0,
        }
        total_waste_mapped = 0

        for det in filtered:
            predicted_label = det["predicted_label"]
            conf = det["confidence"]
            norm_box = det["bounding_box"]

            mapped_info = self.class_mapper.map_detection(
                predicted_label=predicted_label,
                model_type=model_type,
            )

            is_mapped = mapped_info["is_waste_mapped"]
            primary_stream = mapped_info["primary_stream"]

            if is_mapped and primary_stream in stream_counts:
                stream_counts[primary_stream] += 1
                total_waste_mapped += 1

            detected_item = {
                "class_id": mapped_info["mapped_class_id"] or f"unmapped_{predicted_label.replace(' ', '_')}",
                "detected_label": predicted_label,
                "item_name": mapped_info["item_name"] or predicted_label.capitalize(),
                "primary_stream": primary_stream,
                "bin_color": mapped_info["bin_color"],
                "confidence": conf,
                "bounding_box": norm_box,
                "is_waste_mapped": is_mapped,
                "disposal_guidance": mapped_info["disposal_guidance"],
                "segregation_instructions": mapped_info["segregation_instructions"],
                "source_references": mapped_info["source_references"],
                "verification_status": mapped_info["verification_status"],
                "mapping_notes": mapped_info["mapping_notes"],
            }
            detected_items.append(detected_item)

        # Remove zero streams from summary for clean readability
        active_streams = {k: v for k, v in stream_counts.items() if v > 0}

        return {
            "status": "success",
            "message": (
                f"Detection completed. Found {len(detected_items)} object(s), "
                f"{total_waste_mapped} mapped to statutory waste streams."
            ),
            "target_hardware": "Raspberry Pi 5 + AI HAT+ 2 (Hailo-8 / Hailo-8L NPU)",
            "target_model": model_name,
            "pipeline_status": "inference_active",
            "is_integrated": True,
            "model_type": model_type,
            "image_dimensions": {"width": image_width, "height": image_height, "channels": 3},
            "inference_latency_ms": latency_ms,
            "total_detected": len(detected_items),
            "total_waste_mapped": total_waste_mapped,
            "stream_counts": active_streams,
            "detected_items": detected_items,
        }
