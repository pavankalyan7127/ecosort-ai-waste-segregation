"""Health check endpoint route."""

from fastapi import APIRouter
from ...core.config import settings
from ...knowledge.waste_service import WasteKnowledgeService

router = APIRouter(tags=["Health"])
knowledge_service = WasteKnowledgeService()


@router.get("/health", summary="Application & Knowledge Base Health Check")
def get_health():
    """Return application operational status, API version, target hardware,

    and knowledge-base availability.
    """
    total_items = len(knowledge_service.get_all_items())
    is_kb_available = total_items > 0

    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.version,
        "target_hardware": settings.target_hardware,
        "supported_languages": settings.supported_languages,
        "knowledge_base": {
            "status": "available" if is_kb_available else "unavailable",
            "total_items": total_items,
            "verified_items": len(knowledge_service.get_verified_items()),
            "pending_items": len(knowledge_service.get_pending_verification_items()),
            "primary_legislation": knowledge_service.metadata.get(
                "primary_legislation", "Solid Waste Management Rules, 2026"
            ),
        },
    }
