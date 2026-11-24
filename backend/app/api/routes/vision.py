"""Vision object detection route supporting YOLO inference and statutory knowledge base mapping."""

import logging
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ...schemas.vision import VisionDetectResponse
from ...vision.detector import WasteDetector
from ...vision.preprocessing import ImageValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["Computer Vision"])

# Singleton detector instance
_detector = WasteDetector()


@router.post(
    "/detect",
    response_model=VisionDetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Waste Objects and Map to Statutory Streams",
)
def detect_waste_objects(
    file: UploadFile = File(..., description="JPEG/PNG image from camera or user upload"),
    confidence_threshold: float = Form(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for bounding boxes"
    ),
    model_name: Optional[str] = Form(
        default=None,
        description="Optional model filename (defaults to yolo11n.pt)"
    ),
    language: str = Form(default="en", description="Target language code"),
):
    """Detect waste objects in an uploaded image, predict bounding boxes,

    normalize coordinates, and map detected classes to the statutory 4-stream waste taxonomy.

    Authentic behavior:
    - If model weights are not found, reports `status="model_weights_not_found"` without hallucinating detections.
    - COCO demonstration classes are mapped strictly to justifiable statutory waste items.
    - Unrelated classes (e.g. person, car) remain marked as `Unsupported` / `is_waste_mapped=False`.
    """
    try:
        contents = file.file.read()
    except Exception as exc:
        logger.error(f"Failed to read uploaded file stream: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}"
        )

    try:
        result = _detector.detect(
            image_bytes=contents,
            confidence_threshold=confidence_threshold,
            model_name=model_name,
        )
        return result
    except ImageValidationError as exc:
        logger.warning(f"Image validation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )
    except Exception as exc:
        logger.exception(f"Unexpected error during vision detection: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline error: {str(exc)}"
        )
