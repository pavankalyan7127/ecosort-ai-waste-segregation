"""Waste taxonomy and knowledge base retrieval routes."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from ...knowledge.waste_service import (
    WasteKnowledgeService,
    VALID_PRIMARY_STREAMS,
)
from ...schemas.waste import (
    WasteCategoryOverviewSchema,
    WasteItemSchema,
    WasteRecommendationSchema,
    WasteStreamSummarySchema,
)

router = APIRouter(prefix="/waste", tags=["Waste Knowledge Base"])
knowledge_service = WasteKnowledgeService()


@router.get(
    "/categories",
    response_model=List[WasteCategoryOverviewSchema],
    summary="Get Statutory Four-Stream Waste Categories",
)
def get_categories():
    """Return the four primary statutory streams mandated under the Solid Waste Management Rules, 2026:

    - Wet waste (Green)
    - Dry waste (Blue)
    - Sanitary waste (Red / dedicated wrapped)
    - Special care waste (Deposition Centres / EPR)

    Includes relevant statutory descriptions, bin colors, and available item counts.
    """
    return knowledge_service.get_categories_overview()


@router.get(
    "/search",
    response_model=List[WasteItemSchema],
    summary="Search Waste Items by Name or Tamil Alias",
)
def search_waste_items(
    q: str = Query(..., min_length=1, description="Search query in English or Tamil")
):
    """Search knowledge base items by official name, alternative synonyms, or Tamil script aliases.

    Uses deterministic substring matching across verified knowledge entries.
    """
    clean_q = q.strip()
    if not clean_q:
        return []
    return knowledge_service.search_by_name(clean_q)


@router.get(
    "/summary",
    response_model=WasteStreamSummarySchema,
    summary="Get Knowledge Base Four-Stream Summary Statistics",
)
def get_knowledge_summary():
    """Return breakdown of verified and pending entries across all statutory streams."""
    return knowledge_service.get_stream_coverage_summary()


@router.get(
    "/stream/{stream_name}",
    response_model=List[WasteItemSchema],
    summary="Retrieve Items by Statutory Primary Stream",
)
def get_items_by_stream(stream_name: str):
    """Retrieve all waste items belonging to a specified primary stream.

    Valid statutory streams:
    - Wet waste
    - Dry waste
    - Sanitary waste
    - Special care waste

    Returns 400 Bad Request if an invalid stream name is requested.
    """
    normalized = knowledge_service.normalize_stream_name(stream_name)
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid waste stream '{stream_name}'. Valid statutory streams under SWM Rules 2026 are: "
                f"{VALID_PRIMARY_STREAMS}"
            ),
        )

    return knowledge_service.get_items_by_stream(normalized)


@router.get(
    "/recommendation/{class_id}",
    response_model=WasteRecommendationSchema,
    summary="Get Statutory Disposal & Handling Recommendation for an Item",
)
def get_waste_recommendation(class_id: str):
    """Return segregation instructions, preparation steps, handling precautions,

    and disposal facilities for an item, complete with traceable legal source references.
    Flags entries if pending verification.
    """
    rec = knowledge_service.get_handling_and_disposal_recommendation(class_id)
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Waste class ID '{class_id}' not found in the knowledge base.",
        )
    return rec


@router.get(
    "/{class_id}",
    response_model=WasteItemSchema,
    summary="Retrieve Full Waste Item by Unique Class ID",
)
def get_waste_item(class_id: str):
    """Retrieve full item information including primary stream, segregation rules,

    handling precautions, preparation instructions, recycling guidance, environmental details,
    traceable statutory source citations, and verification status.
    """
    item = knowledge_service.get_item_by_class_id(class_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Waste class ID '{class_id}' not found in the knowledge base.",
        )
    return item
