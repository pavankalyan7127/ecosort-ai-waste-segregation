"""Pydantic schemas for multilingual language support endpoint."""

from typing import List
from pydantic import BaseModel


class LanguageDetail(BaseModel):
    code: str
    name: str
    native_name: str
    script: str
    status: str  # "implemented" or "planned"
    capabilities: List[str]  # e.g., ["ui", "knowledge_base", "translation", "tts", "asr"]


class LanguagesResponse(BaseModel):
    implemented_languages: List[LanguageDetail]
    planned_languages: List[LanguageDetail]
    default_language: str
    translation_engine: str
