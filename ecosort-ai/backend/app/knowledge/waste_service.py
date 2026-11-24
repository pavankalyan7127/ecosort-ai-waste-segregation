"""Evidence-Based Waste Knowledge Base Retrieval Service.

Strictly grounded in authentic statutory documents (Solid Waste Management Rules 2026,
E-Waste Rules 2022, Battery Waste Rules 2022, Plastic Waste Rules 2016, BMW Rules 2016,
and Tamil Nadu State Orders).
DOES NOT GENERATE SYNTHETIC OR AI-INVENTED FACTS.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE_BASE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "waste_knowledge_base.json"
)

VALID_PRIMARY_STREAMS = [
    "Wet waste",
    "Dry waste",
    "Sanitary waste",
    "Special care waste",
]


class WasteKnowledgeService:
    """Service to query and retrieve evidence-backed waste taxonomy and handling guidance."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else DEFAULT_KNOWLEDGE_BASE_PATH
        self._metadata: Dict[str, Any] = {}
        self._items_by_id: Dict[str, Dict[str, Any]] = {}
        self._items: List[Dict[str, Any]] = []
        self.load_knowledge_base()

    def load_knowledge_base(self) -> None:
        """Load knowledge base from JSON file."""
        if not self.data_path.exists():
            logger.error(f"Knowledge base file not found at {self.data_path}")
            self._items = []
            self._items_by_id = {}
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._metadata = data.get("metadata", {})
        self._items = data.get("items", [])
        self._items_by_id = {}

        for item in self._items:
            class_id = item.get("class_id")
            if class_id:
                if class_id in self._items_by_id:
                    logger.warning(f"Duplicate class_id detected during load: {class_id}")
                self._items_by_id[class_id] = item

        logger.info(
            f"Loaded {len(self._items)} waste knowledge items from {self.data_path.name}"
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return knowledge base metadata."""
        return self._metadata

    def get_all_items(self) -> List[Dict[str, Any]]:
        """Return all items in the knowledge base."""
        return list(self._items)

    def get_item_by_class_id(self, class_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an item strictly by its unique class_id.

        Returns None if class_id is not found.
        """
        if not class_id:
            return None
        return self._items_by_id.get(class_id.strip())

    def search_by_name(self, query: str) -> List[Dict[str, Any]]:
        """Search items by item_name or alternative_names (case-insensitive substring match).

        Returns matching items from the structured knowledge base without hallucination.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip().lower()
        results = []

        for item in self._items:
            item_name = item.get("item_name", "").lower()
            alt_names = [a.lower() for a in item.get("alternative_names", [])]
            examples = [e.lower() for e in item.get("examples", [])]

            if (
                clean_query in item_name
                or any(clean_query in alt for alt in alt_names)
                or any(clean_query in ex for ex in examples)
            ):
                results.append(item)

        return results

    def normalize_stream_name(self, stream_name: str) -> Optional[str]:
        """Normalize and validate a stream name against the four statutory streams."""
        if not stream_name:
            return None
        clean = stream_name.strip().lower().replace("_", " ").replace("-", " ")
        for valid in VALID_PRIMARY_STREAMS:
            if clean == valid.lower():
                return valid
        return None

    def get_items_by_stream(self, primary_stream: str) -> List[Dict[str, Any]]:
        """Retrieve all items classified under a specific primary stream.

        Valid primary streams in SWM Rules 2026:
        - Wet waste
        - Dry waste
        - Sanitary waste
        - Special care waste
        """
        norm_stream = self.normalize_stream_name(primary_stream)
        if not norm_stream:
            # Try raw comparison if normalization didn't find exact match
            target_stream = primary_stream.strip().lower()
            return [
                item
                for item in self._items
                if item.get("primary_stream", "").lower() == target_stream
            ]

        return [
            item
            for item in self._items
            if item.get("primary_stream") == norm_stream
        ]

    def get_categories_overview(self) -> List[Dict[str, Any]]:
        """Return the four primary statutory streams with descriptions, bin colors,

        legal citations, and item counts from the knowledge base.
        """
        stream_counts = {}
        for item in self._items:
            s = item.get("primary_stream", "Unknown")
            stream_counts[s] = stream_counts.get(s, 0) + 1

        categories = [
            {
                "stream_name": "Wet waste",
                "bin_color": "Green",
                "statutory_definition": "Organic waste including kitchen waste, food waste, vegetable waste, meat waste, fruits waste, flower waste, and such similar waste and biodegradable waste.",
                "legal_citation": "Solid Waste Management Rules, 2026, Rule 3(zzl) & Rule 39(51)",
                "item_count": stream_counts.get("Wet waste", 0),
                "handling_overview": "Segregate at source in green bins without plastic bags. Process via composting, vermi-composting, or bio-methanation.",
                "disposal_facility": "Decentralized Micro Composting Centres (MCCs) or Biomethanation Plants",
            },
            {
                "stream_name": "Dry waste",
                "bin_color": "Blue",
                "statutory_definition": "Waste other than wet waste, sanitary waste, special care waste and includes recyclable waste and non-recyclable waste.",
                "legal_citation": "Solid Waste Management Rules, 2026, Rule 3(s), Rule 3(l) & Rule 39(51)",
                "item_count": stream_counts.get("Dry waste", 0),
                "handling_overview": "Keep clean and dry in blue bins. Segregate into recyclables (paper, plastic, metal, glass) and non-recyclable combustible RDF fraction.",
                "disposal_facility": "Material Recovery Facilities (MRFs), Registered Recyclers, or Cement Kiln Co-Processing",
            },
            {
                "stream_name": "Sanitary waste",
                "bin_color": "Red in public restrooms / Dedicated wrapped storage at home",
                "statutory_definition": "Waste comprising of used diapers, sanitary towels or napkins, tampons, condoms, incontinence sheets and any other similar waste.",
                "legal_citation": "Solid Waste Management Rules, 2026, Rule 3(zp), Rule 5(1)(c) & Rule 39(2)(xii)",
                "item_count": stream_counts.get("Sanitary waste", 0),
                "handling_overview": "Wrap securely in pouches provided by manufacturers or wrapping material instructed by local bodies. Never flush or mix with wet/dry waste.",
                "disposal_facility": "Dedicated High-Temperature Incinerators or Common Bio-medical Waste Treatment Facilities (CBWTF)",
            },
            {
                "stream_name": "Special care waste",
                "bin_color": "Designated Deposition Centre / EPR Collection Kiosk",
                "statutory_definition": "Discarded paint drums, pesticide cans/containers, CFL/tube lights, expired medicines, broken mercury thermometers, waste batteries, needles/syringes, contaminated gauze, and specialized e-waste generated at household level.",
                "legal_citation": "Solid Waste Management Rules, 2026, Rule 3(zx), Rule 39(48); E-Waste Rules 2022; Battery Waste Rules 2022",
                "item_count": stream_counts.get("Special care waste", 0),
                "handling_overview": "Store in separate puncture-proof or leak-proof containers. Hand over to municipal Deposition Centres (1 per 5 sq. km) or authorized take-back centers.",
                "disposal_facility": "Municipal Deposition Centres, Authorized Hazardous Waste Processors, or CBWTF",
            },
        ]
        return categories

    def get_handling_and_disposal_recommendation(
        self, class_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return statutory handling, segregation, and disposal recommendations for an item,

        accompanied by traceable source references.
        Flagged if the entry is pending verification.
        """
        item = self.get_item_by_class_id(class_id)
        if not item:
            return None

        is_pending = item.get("verification_status") != "verified"

        return {
            "class_id": item["class_id"],
            "item_name": item["item_name"],
            "primary_stream": item["primary_stream"],
            "subcategory": item.get("subcategory"),
            "segregation_instructions": item.get("segregation_instructions"),
            "preparation_instructions": item.get("preparation_instructions"),
            "handling_precautions": item.get("handling_precautions"),
            "recycling_or_disposal_guidance": item.get("recycling_or_disposal_guidance"),
            "environmental_information": item.get("environmental_information"),
            "is_pending_verification": is_pending,
            "verification_status": item.get("verification_status", "pending_verification"),
            "source_references": item.get("source_references", []),
        }

    def get_pending_verification_items(self) -> List[Dict[str, Any]]:
        """Return all items that are flagged with pending verification status."""
        return [
            item
            for item in self._items
            if item.get("verification_status") == "pending_verification"
        ]

    def get_verified_items(self) -> List[Dict[str, Any]]:
        """Return all items verified against authentic government sources."""
        return [
            item
            for item in self._items
            if item.get("verification_status") == "verified"
        ]

    def get_stream_coverage_summary(self) -> Dict[str, Any]:
        """Return coverage statistics by primary stream and verification status."""
        summary: Dict[str, Dict[str, int]] = {}
        for item in self._items:
            stream = item.get("primary_stream", "Unknown")
            status = item.get("verification_status", "pending_verification")
            if stream not in summary:
                summary[stream] = {"total": 0, "verified": 0, "pending": 0}
            summary[stream]["total"] += 1
            if status == "verified":
                summary[stream]["verified"] += 1
            else:
                summary[stream]["pending"] += 1

        return {
            "total_items": len(self._items),
            "total_verified": len(self.get_verified_items()),
            "total_pending": len(self.get_pending_verification_items()),
            "streams": summary,
        }
