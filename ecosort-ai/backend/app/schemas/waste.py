"""Pydantic schemas for waste knowledge base endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field


class SourceReferenceSchema(BaseModel):
    document_title: str
    issuing_organisation: str
    official_url: str
    rule_or_section: str
    page_number: str
    publication_or_notification_date: Optional[str] = None


class WasteItemSchema(BaseModel):
    class_id: str
    item_name: str
    alternative_names: List[str] = []
    primary_stream: str
    subcategory: Optional[str] = None
    description: str
    examples: List[str] = []
    segregation_instructions: str
    preparation_instructions: Optional[str] = None
    handling_precautions: Optional[str] = None
    recycling_or_disposal_guidance: Optional[str] = None
    environmental_information: Optional[str] = None
    source_references: List[SourceReferenceSchema] = []
    verification_status: str = Field(
        ..., description="'verified' or 'pending_verification'"
    )


class WasteCategoryOverviewSchema(BaseModel):
    stream_name: str
    bin_color: str
    statutory_definition: str
    legal_citation: str
    item_count: int
    handling_overview: str
    disposal_facility: str


class WasteRecommendationSchema(BaseModel):
    class_id: str
    item_name: str
    primary_stream: str
    subcategory: Optional[str] = None
    segregation_instructions: Optional[str] = None
    preparation_instructions: Optional[str] = None
    handling_precautions: Optional[str] = None
    recycling_or_disposal_guidance: Optional[str] = None
    environmental_information: Optional[str] = None
    is_pending_verification: bool
    verification_status: str
    source_references: List[SourceReferenceSchema] = []


class WasteStreamSummarySchema(BaseModel):
    total_items: int
    total_verified: int
    total_pending: int
    streams: dict
