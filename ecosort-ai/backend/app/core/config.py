"""Core configuration and settings for EcoSort AI."""

import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load optional .env file
load_dotenv()

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BACKEND_ROOT / "data"
MODELS_DIR = BACKEND_ROOT / "models"


class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "EcoSort AI")
    version: str = "0.1.0"
    description: str = (
        "AI-Powered Multilingual Multimodal Standalone Waste Segregation Kiosk"
    )
    api_prefix: str = "/api"
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    waste_rules_file: Path = DATA_DIR / "waste_rules.json"

    # Languages
    supported_languages: list[str] = ["en", "ta"]
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "en")

    # Hardware target
    target_hardware: str = os.getenv("TARGET_HARDWARE", "Raspberry Pi 5 + AI HAT+ 2")


settings = Settings()
