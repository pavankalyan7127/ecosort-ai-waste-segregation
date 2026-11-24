"""Pydantic schemas for conversational assistant endpoint contract."""

from typing import List, Optional
from pydantic import BaseModel, Field


class AssistantChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Citizen inquiry or waste sorting question")
    language: str = Field(default="en", description="Target ISO 639-1 language code ('en' or 'ta')")
    detected_items: Optional[List[str]] = Field(
        default=None, description="Optional class_ids of waste objects currently in vision frame"
    )


class AssistantChatResponse(BaseModel):
    status: str = Field(
        default="pending_model_integration",
        description="Assistant status: 'pending_model_integration' or 'success'"
    )
    message: str
    target_hardware: str
    target_model: str
    runtime: str
    language: str
    is_integrated: bool = False
    response: Optional[str] = None
