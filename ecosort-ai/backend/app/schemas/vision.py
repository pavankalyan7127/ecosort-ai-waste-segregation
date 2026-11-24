"""Pydantic schemas for vision detection endpoint contract and inference results."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates bounded between 0.0 and 1.0."""
    x_min: float = Field(..., ge=0.0, le=1.0, description="Normalized minimum X coordinate")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Normalized minimum Y coordinate")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Normalized maximum X coordinate")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Normalized maximum Y coordinate")


class ImageDimensions(BaseModel):
    """Dimensions of the processed input frame."""
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    channels: int = Field(default=3, description="Image color channels (RGB)")


class DetectedWasteItemContract(BaseModel):
    """Enriched detection entry mapped to statutory waste taxonomy."""
    class_id: str = Field(..., description="Statutory knowledge base ID or unmapped identifier")
    detected_label: str = Field(..., description="Raw label predicted by YOLO model")
    item_name: Optional[str] = Field(None, description="Standard statutory item name")
    primary_stream: str = Field(..., description="One of Wet waste, Dry waste, Sanitary waste, Special care waste, or Unsupported")
    bin_color: Optional[str] = Field(None, description="Recommended bin color (Green/Blue/Red/Deposition)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bounding_box: BoundingBox = Field(..., description="Normalized bounding box")
    is_waste_mapped: bool = Field(default=True, description="True if genuinely mapped to statutory taxonomy")
    disposal_guidance: Optional[str] = Field(None, description="Statutory disposal directive")
    segregation_instructions: Optional[str] = Field(None, description="Preparation and segregation steps")
    source_references: List[Dict[str, Any]] = Field(default_factory=list, description="Authentic legal/regulatory citations")
    verification_status: Optional[str] = Field(None, description="Statutory verification level")
    mapping_notes: Optional[str] = Field(None, description="Explanation of label-to-stream mapping")


class VisionDetectResponse(BaseModel):
    """Response payload for POST /api/vision/detect."""
    status: str = Field(
        ...,
        description="Pipeline execution status: 'success', 'model_weights_not_found', or 'error'"
    )
    message: str = Field(..., description="Human-readable status summary")
    target_hardware: str = Field(
        default="Raspberry Pi 5 + AI HAT+ 2 (Hailo-8 / Hailo-8L NPU)",
        description="Target deployment hardware platform"
    )
    target_model: str = Field(..., description="Computer vision model name")
    pipeline_status: str = Field(..., description="Internal pipeline status code")
    is_integrated: bool = Field(default=True, description="True if modular vision pipeline is integrated")
    model_type: Optional[str] = Field(None, description="'coco_demonstration' | 'waste_specific_finetuned' | 'unavailable'")
    image_dimensions: Optional[ImageDimensions] = Field(None, description="Decoded image dimensions")
    inference_latency_ms: float = Field(default=0.0, description="Inference latency in milliseconds")
    total_detected: int = Field(default=0, description="Total detected objects passing threshold")
    total_waste_mapped: int = Field(default=0, description="Count of objects mapped to statutory waste streams")
    stream_counts: Dict[str, int] = Field(default_factory=dict, description="Counts of objects by primary waste stream")
    detected_items: List[DetectedWasteItemContract] = Field(default_factory=list, description="List of detected and mapped objects")
