"""Central API router aggregating modular sub-routers."""

from fastapi import APIRouter
from .routes import health, waste, vision, assistant, languages
from ..translation import MultilingualTranslator

api_router = APIRouter()
translator = MultilingualTranslator()

# Include modular route handlers
api_router.include_router(health.router)
api_router.include_router(waste.router)
api_router.include_router(vision.router)
api_router.include_router(assistant.router)
api_router.include_router(languages.router)

# Backward-compatibility alias for existing frontend components:
# Allows frontend getCategories() at /api/knowledge/categories to resolve smoothly
@api_router.get(
    "/knowledge/categories",
    tags=["Waste Knowledge Base (Compatibility)"],
    include_in_schema=False,
)
def get_categories_compat():
    """Compatibility alias pointing to /api/waste/categories."""
    from .routes.waste import get_categories
    return get_categories()


@api_router.post(
    "/translation/translate",
    tags=["Languages & Localization (Compatibility)"],
    include_in_schema=False,
)
def translate_text(request: dict):
    """Compatibility endpoint for basic translation."""
    text = request.get("text", "")
    src_lang = request.get("src_lang", "en")
    tgt_lang = request.get("tgt_lang", "ta")
    translated = translator.translate(text=text, src_lang=src_lang, tgt_lang=tgt_lang)
    return {
        "original_text": text,
        "translated_text": translated,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    }
