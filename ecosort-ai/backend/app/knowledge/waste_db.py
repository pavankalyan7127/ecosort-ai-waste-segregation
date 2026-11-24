"""Local waste knowledge base and segregation rule provider."""

import json
from pathlib import Path
from typing import Any, Optional
from ..core.config import settings


class WasteKnowledgeBase:
    """Provides local rule-based waste segregation guidance."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or settings.waste_rules_file
        self._categories: list[dict[str, Any]] = []
        self.load_rules()

    def load_rules(self) -> None:
        if self.data_path.exists():
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._categories = data.get("categories", [])
        else:
            self._categories = []

    def get_all_categories(self, lang: str = "en") -> list[dict[str, Any]]:
        """Return all waste categories localized to the requested language."""
        results = []
        for cat in self._categories:
            results.append({
                "id": cat.get("id"),
                "bin_color": cat.get("bin_color"),
                "name": cat.get("name", {}).get(lang, cat.get("name", {}).get("en", "")),
                "description": cat.get("description", {}).get(lang, cat.get("description", {}).get("en", "")),
                "disposal_instructions": cat.get("disposal_instructions", {}).get(
                    lang, cat.get("disposal_instructions", {}).get("en", "")
                ),
                "examples": cat.get("examples", [])
            })
        return results

    def find_category_by_item(self, item_label: str) -> Optional[dict[str, Any]]:
        """Find category matching a detected item label."""
        clean_item = item_label.lower().strip()
        for cat in self._categories:
            if clean_item in cat.get("examples", []) or clean_item == cat.get("id"):
                return cat
        return None
