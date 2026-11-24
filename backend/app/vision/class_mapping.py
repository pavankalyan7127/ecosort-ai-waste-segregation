"""Class mapping layer bridging computer vision detection labels with the statutory waste knowledge base."""

import logging
from typing import Any, Dict, Optional
from ..knowledge.waste_service import WasteKnowledgeService

logger = logging.getLogger(__name__)

# Genuinely justifiable mappings for COCO pretrained demonstration classes.
# STRICT DIRECTIVE: Unrelated classes (e.g., person, car, dog, chair) are NEVER mapped to waste.
COCO_DEMONSTRATION_MAPPING: Dict[str, Dict[str, str]] = {
    "bottle": {
        "class_id": "dry_plastic_pet_bottles",
        "notes": "Demonstration mapping: COCO 'bottle' mapped to statutory Dry waste (PET bottle).",
    },
    "cup": {
        "class_id": "dry_plastic_pp_containers",
        "notes": "Demonstration mapping: COCO 'cup' mapped to statutory Dry waste (plastic/disposable container).",
    },
    "banana": {
        "class_id": "wet_fruit_vegetable_peels",
        "notes": "Demonstration mapping: COCO 'banana' mapped to statutory Wet waste (organic fruit scrap).",
    },
    "apple": {
        "class_id": "wet_fruit_vegetable_peels",
        "notes": "Demonstration mapping: COCO 'apple' mapped to statutory Wet waste (organic fruit scrap).",
    },
    "orange": {
        "class_id": "wet_fruit_vegetable_peels",
        "notes": "Demonstration mapping: COCO 'orange' mapped to statutory Wet waste (organic fruit scrap).",
    },
    "broccoli": {
        "class_id": "wet_fruit_vegetable_peels",
        "notes": "Demonstration mapping: COCO 'broccoli' mapped to statutory Wet waste (vegetable scrap).",
    },
    "carrot": {
        "class_id": "wet_fruit_vegetable_peels",
        "notes": "Demonstration mapping: COCO 'carrot' mapped to statutory Wet waste (vegetable scrap).",
    },
    "pizza": {
        "class_id": "wet_kitchen_food_scraps",
        "notes": "Demonstration mapping: COCO 'pizza' mapped to statutory Wet waste (cooked meal leftovers).",
    },
    "donut": {
        "class_id": "wet_kitchen_food_scraps",
        "notes": "Demonstration mapping: COCO 'donut' mapped to statutory Wet waste (food scrap).",
    },
    "cake": {
        "class_id": "wet_kitchen_food_scraps",
        "notes": "Demonstration mapping: COCO 'cake' mapped to statutory Wet waste (food scrap).",
    },
    "hot dog": {
        "class_id": "wet_kitchen_food_scraps",
        "notes": "Demonstration mapping: COCO 'hot dog' mapped to statutory Wet waste (cooked meal leftovers).",
    },
    "book": {
        "class_id": "dry_paper_newspaper",
        "notes": "Demonstration mapping: COCO 'book' mapped to statutory Dry waste (paper/cellulose).",
    },
    "cell phone": {
        "class_id": "special_ewaste_smartphones_tablets",
        "notes": "Demonstration mapping: COCO 'cell phone' mapped to statutory Special care waste (E-Waste Rules 2022).",
    },
}

# Fine-tuned waste detection model classes (e.g., TACO & custom fine-tuned weights)
WASTE_SPECIFIC_MAPPING: Dict[str, Dict[str, str]] = {
    "clear plastic bottle": {"class_id": "dry_plastic_pet_bottles", "notes": "TACO clear bottle"},
    "other plastic bottle": {"class_id": "dry_plastic_hdpe_containers", "notes": "TACO opaque bottle"},
    "plastic bottle cap": {"class_id": "dry_plastic_pp_containers", "notes": "TACO bottle cap"},
    "drink can": {"class_id": "dry_metal_aluminium_cans", "notes": "TACO aluminium beverage can"},
    "food can": {"class_id": "dry_metal_steel_tin_cans", "notes": "TACO steel food tin"},
    "drink carton": {"class_id": "dry_paper_tetra_packs", "notes": "TACO aseptic composite carton"},
    "corrugated carton": {"class_id": "dry_paper_cardboard_cartons", "notes": "TACO cardboard box"},
    "normal paper": {"class_id": "dry_paper_newspaper", "notes": "TACO paper scrap"},
    "glass bottle": {"class_id": "dry_glass_bottles_jars", "notes": "TACO intact glass bottle"},
    "broken glass": {"class_id": "dry_glass_broken_shards", "notes": "TACO glass shards"},
    "crisp packet": {"class_id": "dry_plastic_mlp_non_recyclable", "notes": "TACO multi-layer snack pouch"},
    "plastic film": {"class_id": "dry_plastic_ldpe_films", "notes": "TACO LDPE film packaging"},
    "plastic bag": {"class_id": "dry_plastic_single_use_banned", "notes": "TACO plastic carry bag"},
    "battery": {"class_id": "special_batteries_household_dry_cell", "notes": "TACO portable battery"},
    "cfl": {"class_id": "special_cfl_tube_lights", "notes": "Hazardous mercury lighting"},
    "syringe": {"class_id": "special_needles_syringes_sharps", "notes": "Medical sharps"},
}


class WasteClassMapper:
    """Bridges model class labels to the statutory 42-item waste knowledge base."""

    def __init__(self, knowledge_service: Optional[WasteKnowledgeService] = None):
        self.knowledge_service = knowledge_service or WasteKnowledgeService()

    def get_supported_classes(self, model_type: str = "coco_demonstration") -> list:
        """Return list of model labels supported by the given mapping profile."""
        if model_type == "waste_specific_finetuned":
            return list(WASTE_SPECIFIC_MAPPING.keys())
        return list(COCO_DEMONSTRATION_MAPPING.keys())

    def map_detection(
        self,
        predicted_label: str,
        model_type: str = "coco_demonstration",
    ) -> Dict[str, Any]:
        """Map a model label to the knowledge base and return enriched statutory guidance.

        If unmapped, returns is_waste_mapped=False and marks stream as Unsupported.
        NEVER FABRICATES UNVERIFIED DISPOSAL DIRECTIVES.
        """
        clean_label = predicted_label.strip().lower()

        # Select appropriate mapping dictionary
        mapping_dict = (
            WASTE_SPECIFIC_MAPPING
            if model_type == "waste_specific_finetuned"
            else COCO_DEMONSTRATION_MAPPING
        )

        mapping_info = mapping_dict.get(clean_label)

        if not mapping_info:
            return {
                "is_waste_mapped": False,
                "mapped_class_id": None,
                "item_name": None,
                "primary_stream": "Unsupported",
                "bin_color": None,
                "disposal_guidance": None,
                "segregation_instructions": None,
                "source_references": [],
                "verification_status": None,
                "mapping_notes": f"Label '{predicted_label}' has no statutory waste taxonomy mapping in profile '{model_type}'.",
            }

        target_class_id = mapping_info["class_id"]
        kb_item = self.knowledge_service.get_item_by_class_id(target_class_id)

        if not kb_item:
            logger.warning(f"Target class_id '{target_class_id}' not found in knowledge base!")
            return {
                "is_waste_mapped": False,
                "mapped_class_id": target_class_id,
                "item_name": None,
                "primary_stream": "Unknown",
                "bin_color": None,
                "disposal_guidance": None,
                "segregation_instructions": None,
                "source_references": [],
                "verification_status": None,
                "mapping_notes": f"Mapped class_id '{target_class_id}' missing from knowledge base.",
            }

        bin_color = "Green" if kb_item["primary_stream"] == "Wet waste" else (
            "Blue" if kb_item["primary_stream"] == "Dry waste" else (
                "Red (public) / Pouch-wrapped" if kb_item["primary_stream"] == "Sanitary waste" else "Deposition Centre"
            )
        )

        return {
            "is_waste_mapped": True,
            "mapped_class_id": kb_item["class_id"],
            "item_name": kb_item["item_name"],
            "primary_stream": kb_item["primary_stream"],
            "bin_color": bin_color,
            "disposal_guidance": kb_item.get("recycling_or_disposal_guidance"),
            "segregation_instructions": kb_item.get("segregation_instructions"),
            "source_references": kb_item.get("source_references", []),
            "verification_status": kb_item.get("verification_status", "verified"),
            "mapping_notes": mapping_info.get("notes", "Statutory mapping verified."),
        }
