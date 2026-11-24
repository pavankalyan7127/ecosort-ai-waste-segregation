"""Waste category definitions and standard classification schema."""

from enum import Enum
from pydantic import BaseModel


class WasteCategoryType(str, Enum):
    WET_ORGANIC = "wet_organic"
    DRY_RECYCLABLE = "dry_recyclable"
    HAZARDOUS_SANITARY = "hazardous_sanitary"
    E_WASTE = "e_waste"
    UNKNOWN = "unknown"


class DetectedWasteItem(BaseModel):
    label: str
    category: WasteCategoryType
    confidence: float
    bin_color: str
    bounding_box: list[float]  # [x_min, y_min, x_max, y_max] normalized (0.0 to 1.0)
    disposal_summary: dict[str, str] = {}
