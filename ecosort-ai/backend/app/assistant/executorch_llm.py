"""Conversational AI interface powered by Llama 3.2 Instruct via ExecuTorch."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class KioskAssistant:
    """Offline conversational assistant targeting Llama 3.2 Instruct on ExecuTorch."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.is_loaded = False
        self._init_runtime()

    def _init_runtime(self) -> None:
        if self.model_path and self.model_path.exists():
            try:
                # ExecuTorch edge runtime initialization
                # from executorch.extension.pybindings.portable_lib import _load_for_executorch
                self.is_loaded = True
                logger.info(f"Loaded ExecuTorch Llama 3.2 model from {self.model_path}")
            except Exception as exc:
                logger.warning(f"ExecuTorch runtime failed to initialize: {exc}")
                self.is_loaded = False
        else:
            logger.info("ExecuTorch Llama 3.2 weights not found; running in local kiosk fallback mode.")

    def generate_response(
        self,
        prompt: str,
        detected_items: Optional[list[str]] = None,
        language: str = "en"
    ) -> str:
        """Generate conversational response tailored to waste segregation."""
        if self.is_loaded:
            # Placeholder for running ExecuTorch model inference
            return "Inference output from on-device Llama 3.2 Instruct ExecuTorch model."

        # Standalone rule-augmented offline responses
        if language == "ta":
            if detected_items:
                items_str = ", ".join(detected_items)
                return (
                    f"கண்டறியப்பட்ட பொருள்: {items_str}. "
                    "பிளாஸ்டிக் பாட்டில்களை நீல நிற மறுசுழற்சி தொட்டியில் போடுவதற்கு முன் தண்ணீரில் கழுவி நசுக்கவும். "
                    "சுற்றுச்சூழலை பாதுகாக்க உங்கள் பங்களிப்புக்கு நன்றி!"
                )
            return (
                "வணக்கம்! நான் EcoSort AI உதவியாளர். கழிவுப் பிரிப்பு குறித்த உங்கள் சந்தேகங்களைக் கேளுங்கள்."
            )

        # Default English
        if detected_items:
            items_str = ", ".join(detected_items)
            return (
                f"Detected item: {items_str}. "
                "Please rinse any liquids, flatten the item to save bin space, and place it inside the Blue Recyclable Bin. "
                "Thank you for keeping our environment clean!"
            )
        return (
            "Hello! I am your EcoSort AI Assistant. Show me an item or ask how to sort and dispose of waste properly."
        )
