"""Model loading, lifecycle management, and caching for YOLO computer vision models."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Primary external resource storage directories on D: drive
EXTERNAL_RESOURCE_ROOT = Path("D:/EcoSort-Resources")
DEFAULT_PRETRAINED_DIR = EXTERNAL_RESOURCE_ROOT / "models" / "pretrained"
DEFAULT_FINETUNED_DIR = EXTERNAL_RESOURCE_ROOT / "models" / "fine-tuned"
LOCAL_FALLBACK_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class ModelManager:
    """Manages loading, singleton caching, and status tracking for YOLO vision models."""

    _instance: Optional["ModelManager"] = None
    _cached_models: Dict[str, Any] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Avoid re-initialization in singleton
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.default_model_name = "yolo11n_waste.pt"
        self.fallback_model_name = "yolo11n.pt"

    def find_model_path(self, model_name: Optional[str] = None) -> Optional[Path]:
        """Locate model weights across configured external and local candidate paths."""
        targets = [model_name] if model_name else [self.default_model_name, self.fallback_model_name]

        for target_name in targets:
            candidate_paths = [
                DEFAULT_FINETUNED_DIR / target_name,
                LOCAL_FALLBACK_DIR / target_name,
                DEFAULT_PRETRAINED_DIR / target_name,
                Path(target_name),  # current directory or direct path
            ]

            for p in candidate_paths:
                if p.is_file():
                    return p

        return None

    def get_model(
        self,
        model_name: Optional[str] = None,
        force_reload: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve or load a YOLO model instance.

        Returns a dictionary containing:
        - 'model': YOLO instance or None
        - 'is_loaded': bool
        - 'status': 'loaded' | 'model_weights_not_found' | 'error'
        - 'model_path': Path or None
        - 'model_type': 'waste_specific_finetuned' | 'coco_demonstration' | 'unavailable'
        - 'error_message': Optional error string
        """
        target_name = model_name or self.default_model_name
        cache_key = str(target_name)

        if not force_reload and cache_key in self._cached_models:
            return self._cached_models[cache_key]

        model_path = self.find_model_path(target_name)

        if not model_path:
            result = {
                "model": None,
                "is_loaded": False,
                "status": "model_weights_not_found",
                "model_name": target_name,
                "model_path": str(DEFAULT_PRETRAINED_DIR / target_name),
                "model_type": "unavailable",
                "error_message": (
                    f"Model weights '{target_name}' not found. "
                    f"Expected location: '{DEFAULT_PRETRAINED_DIR / target_name}'. "
                    "Download confirmation required before downloading large weights."
                ),
            }
            self._cached_models[cache_key] = result
            return result

        try:
            logger.info(f"Loading YOLO model weights from {model_path}...")
            from ultralytics import YOLO

            model = YOLO(str(model_path))

            # Distinguish waste-specific fine-tuned model from standard COCO pretrained model
            is_waste_specific = "taco" in model_path.name.lower() or "waste" in model_path.name.lower() or "fine-tuned" in str(model_path).lower()
            model_type = "waste_specific_finetuned" if is_waste_specific else "coco_demonstration"

            result = {
                "model": model,
                "is_loaded": True,
                "status": "loaded",
                "model_name": model_path.name,
                "model_path": str(model_path),
                "model_type": model_type,
                "error_message": None,
            }
            self._cached_models[cache_key] = result
            logger.info(f"Successfully loaded {model_path.name} ({model_type})")
            return result
        except Exception as exc:
            logger.error(f"Error loading YOLO weights from {model_path}: {exc}")
            result = {
                "model": None,
                "is_loaded": False,
                "status": "error",
                "model_name": target_name,
                "model_path": str(model_path),
                "model_type": "unavailable",
                "error_message": f"Failed to initialize Ultralytics model: {exc}",
            }
            self._cached_models[cache_key] = result
            return result

    def clear_cache(self) -> None:
        """Clear cached model instances to free memory."""
        self._cached_models.clear()
