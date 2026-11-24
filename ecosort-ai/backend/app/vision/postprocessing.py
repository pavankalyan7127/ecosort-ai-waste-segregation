"""Post-processing utilities for object detection bounding boxes and confidence filtering."""

from typing import Any, Dict, List, Tuple


def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp float to specified range [min_val, max_val]."""
    return max(min_val, min(val, max_val))


def normalize_bounding_box(
    box_pixels: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Dict[str, float]:
    """Convert absolute pixel coordinates [x1, y1, x2, y2] to normalized [0.0, 1.0] coordinates

    with clamping to guarantee bounding box is inside frame boundaries.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image dimensions: {image_width}x{image_height}")

    x1, y1, x2, y2 = box_pixels

    # Ensure x_min < x_max and y_min < y_max
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    norm_x_min = clamp(round(x_min / image_width, 4))
    norm_y_min = clamp(round(y_min / image_height, 4))
    norm_x_max = clamp(round(x_max / image_width, 4))
    norm_y_max = clamp(round(y_max / image_height, 4))

    return {
        "x_min": norm_x_min,
        "y_min": norm_y_min,
        "x_max": norm_x_max,
        "y_max": norm_y_max,
    }


def filter_and_format_detections(
    raw_detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    confidence_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """Filter raw detection boxes by confidence threshold, normalize coordinates,

    and sort by confidence in descending order.
    """
    filtered = []
    for det in raw_detections:
        conf = float(det.get("confidence", 0.0))
        if conf < confidence_threshold:
            continue

        raw_box = det.get("box_pixels")
        if not raw_box or len(raw_box) != 4:
            continue

        norm_box = normalize_bounding_box(raw_box, image_width, image_height)

        filtered.append({
            "predicted_label": det.get("label", "unknown"),
            "confidence": round(conf, 4),
            "bounding_box": norm_box,
        })

    # Sort highest confidence first
    filtered.sort(key=lambda x: x["confidence"], reverse=True)
    return filtered
