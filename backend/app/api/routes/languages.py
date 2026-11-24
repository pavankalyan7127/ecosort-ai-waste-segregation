"""Multilingual capabilities route."""

from fastapi import APIRouter
from ...schemas.languages import LanguageDetail, LanguagesResponse

router = APIRouter(prefix="/languages", tags=["Languages & Localization"])

IMPLEMENTED_LANGUAGES = [
    LanguageDetail(
        code="en",
        name="English",
        native_name="English",
        script="Latin",
        status="implemented",
        capabilities=["ui", "knowledge_base", "rest_api", "disposal_directives"],
    ),
    LanguageDetail(
        code="ta",
        name="Tamil",
        native_name="தமிழ்",
        script="Tamil",
        status="implemented",
        capabilities=["ui", "knowledge_base", "search_aliases", "disposal_directives"],
    ),
]

PLANNED_LANGUAGES = [
    LanguageDetail(
        code="hi",
        name="Hindi",
        native_name="हिन्दी",
        script="Devanagari",
        status="planned",
        capabilities=["indictrans2_translation", "indic_conformer_asr", "indic_tts"],
    ),
    LanguageDetail(
        code="te",
        name="Telugu",
        native_name="తెలుగు",
        script="Telugu",
        status="planned",
        capabilities=["indictrans2_translation", "indic_tts"],
    ),
    LanguageDetail(
        code="kn",
        name="Kannada",
        native_name="ಕನ್ನಡ",
        script="Kannada",
        status="planned",
        capabilities=["indictrans2_translation", "indic_tts"],
    ),
    LanguageDetail(
        code="ml",
        name="Malayalam",
        native_name="മലയാളം",
        script="Malayalam",
        status="planned",
        capabilities=["indictrans2_translation", "indic_tts"],
    ),
]


@router.get(
    "",
    response_model=LanguagesResponse,
    summary="Get Supported and Planned Application Languages",
)
def get_supported_languages():
    """Return all languages supported by EcoSort AI, distinguishing currently implemented

    languages (English, Tamil) from planned Indian languages for IndicTrans2 / ASR / TTS.
    """
    return LanguagesResponse(
        implemented_languages=IMPLEMENTED_LANGUAGES,
        planned_languages=PLANNED_LANGUAGES,
        default_language="en",
        translation_engine="AI4Bharat IndicTrans2 (Planned Integration)",
    )
