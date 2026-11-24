"""Automated Validation Test Suite for EcoSort AI Waste Knowledge Base.

Validates:
1. Primary stream conformance (Wet waste, Dry waste, Sanitary waste, Special care waste).
2. Traceability of every factual entry to authentic government sources.
3. Completeness of source reference metadata (title, issuing organisation, rule, page).
4. No unsupported disposal recommendations.
5. Explicit representation of missing information (None / null).
6. Uniqueness of class IDs (duplicate rejection).
7. Identifiability of pending verification entries.
8. Retrieval service operation without synthetic fact generation.
"""

import json
from pathlib import Path
import pytest

from app.knowledge.waste_service import (
    WasteKnowledgeService,
    DEFAULT_KNOWLEDGE_BASE_PATH,
)

VALID_PRIMARY_STREAMS = {
    "Wet waste",
    "Dry waste",
    "Sanitary waste",
    "Special care waste",
}


@pytest.fixture(scope="module")
def raw_data():
    """Load raw JSON data directly to validate storage schema."""
    assert DEFAULT_KNOWLEDGE_BASE_PATH.exists(), (
        f"Knowledge base file missing at {DEFAULT_KNOWLEDGE_BASE_PATH}"
    )
    with open(DEFAULT_KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="module")
def service():
    """Instantiate knowledge retrieval service."""
    return WasteKnowledgeService(DEFAULT_KNOWLEDGE_BASE_PATH)


# --------------------------------------------------------------------
# 1. Structural & Integrity Tests
# --------------------------------------------------------------------

def test_metadata_present(raw_data):
    """Verify metadata block contains statutory references and counts."""
    metadata = raw_data.get("metadata", {})
    assert "Solid Waste Management Rules, 2026" in metadata.get("primary_legislation", "")
    assert metadata.get("total_entries", 0) > 0
    assert metadata.get("verified_entries", 0) > 0
    assert metadata.get("pending_entries", 0) >= 0


def test_no_duplicate_class_ids(raw_data):
    """Validate that every entry has a strictly unique class_id."""
    items = raw_data.get("items", [])
    class_ids = [item.get("class_id") for item in items]
    assert len(class_ids) == len(set(class_ids)), "Duplicate class_id detected in knowledge base!"


# --------------------------------------------------------------------
# 2. Four-Stream Conformance Tests
# --------------------------------------------------------------------

def test_every_entry_has_valid_primary_stream(raw_data):
    """Verify every entry conforms to one of the four statutory streams."""
    items = raw_data.get("items", [])
    for item in items:
        stream = item.get("primary_stream")
        assert stream in VALID_PRIMARY_STREAMS, (
            f"Item {item.get('class_id')} has invalid primary_stream: '{stream}'. "
            f"Must be one of {VALID_PRIMARY_STREAMS}"
        )


def test_four_stream_coverage(service):
    """Verify that all four statutory streams are represented."""
    summary = service.get_stream_coverage_summary()
    streams = summary.get("streams", {})
    for stream in VALID_PRIMARY_STREAMS:
        assert stream in streams, f"Statutory stream '{stream}' has zero entries in knowledge base!"
        assert streams[stream]["total"] > 0, f"Stream '{stream}' has zero entries!"
        assert streams[stream]["verified"] > 0, f"Stream '{stream}' has no verified entries!"


# --------------------------------------------------------------------
# 3. Source Traceability & Grounding Tests
# --------------------------------------------------------------------

def test_every_entry_has_source_references(raw_data):
    """Ensure every factual entry has identifiable source references."""
    items = raw_data.get("items", [])
    for item in items:
        refs = item.get("source_references")
        assert isinstance(refs, list), f"Item {item.get('class_id')} missing source_references list"
        assert len(refs) > 0, (
            f"STRICT RULE VIOLATION: Item {item.get('class_id')} has no source references!"
        )


def test_source_references_completeness(raw_data):
    """Ensure every source reference has required metadata fields."""
    items = raw_data.get("items", [])
    for item in items:
        for ref in item.get("source_references", []):
            assert ref.get("document_title"), (
                f"Item {item.get('class_id')} has reference missing document_title"
            )
            assert ref.get("issuing_organisation"), (
                f"Item {item.get('class_id')} has reference missing issuing_organisation"
            )
            assert ref.get("rule_or_section"), (
                f"Item {item.get('class_id')} has reference missing rule_or_section"
            )
            assert "page_number" in ref, (
                f"Item {item.get('class_id')} has reference missing page_number"
            )


def test_no_unsupported_disposal_recommendations(raw_data):
    """Verify no entry provides a disposal recommendation without source backing."""
    items = raw_data.get("items", [])
    for item in items:
        disposal = item.get("recycling_or_disposal_guidance")
        if disposal:
            refs = item.get("source_references", [])
            assert len(refs) > 0, (
                f"Unsupported disposal recommendation for {item.get('class_id')} without sources!"
            )


def test_missing_information_is_explicit(raw_data):
    """Verify that unverified or unavailable information is explicitly represented as None (null)."""
    items = raw_data.get("items", [])
    pending_items = [i for i in items if i.get("verification_status") == "pending_verification"]
    assert len(pending_items) > 0, "Expected test pending items for validation."
    for item in pending_items:
        # At least one field should be explicitly None or null
        has_explicit_none = any(
            item.get(field) is None
            for field in [
                "preparation_instructions",
                "handling_precautions",
                "recycling_or_disposal_guidance",
                "environmental_information",
            ]
        )
        assert has_explicit_none, (
            f"Pending item {item.get('class_id')} should explicitly represent missing fields as null/None."
        )


# --------------------------------------------------------------------
# 4. Verification Status & Identifiability Tests
# --------------------------------------------------------------------

def test_pending_entries_are_identifiable(service):
    """Verify that pending entries can be retrieved and flagged."""
    pending = service.get_pending_verification_items()
    assert len(pending) > 0
    for item in pending:
        rec = service.get_handling_and_disposal_recommendation(item["class_id"])
        assert rec is not None
        assert rec["is_pending_verification"] is True
        assert rec["verification_status"] == "pending_verification"


def test_verified_entries_are_clean(service):
    """Verify that verified entries have complete recommendations and sources."""
    verified = service.get_verified_items()
    assert len(verified) >= 40
    for item in verified:
        rec = service.get_handling_and_disposal_recommendation(item["class_id"])
        assert rec is not None
        assert rec["is_pending_verification"] is False
        assert rec["verification_status"] == "verified"
        assert rec["segregation_instructions"] is not None
        assert len(rec["source_references"]) > 0


# --------------------------------------------------------------------
# 5. Retrieval Service Functional Tests
# --------------------------------------------------------------------

def test_retrieve_by_class_id(service):
    """Test retrieval by class_id."""
    item = service.get_item_by_class_id("wet_kitchen_food_scraps")
    assert item is not None
    assert item["item_name"] == "Cooked Food Scraps and Meal Leftovers"
    assert item["primary_stream"] == "Wet waste"

    # Non-existent ID returns None without exception
    assert service.get_item_by_class_id("non_existent_waste_item_xyz") is None


def test_search_by_name_and_tamil_alias(service):
    """Test searching by English name, alternative name, and Tamil alias."""
    # Search English
    results_en = service.search_by_name("battery")
    assert len(results_en) >= 2
    assert any("batter" in item["class_id"] for item in results_en)

    # Search Tamil alias
    results_ta = service.search_by_name("பாட்டில்")  # Tamil for bottle
    assert len(results_ta) >= 1

    # Empty search returns empty list
    assert service.search_by_name("") == []
    assert service.search_by_name("   ") == []


def test_retrieve_by_stream(service):
    """Test stream-level retrieval."""
    sanitary_items = service.get_items_by_stream("Sanitary waste")
    assert len(sanitary_items) >= 5
    for item in sanitary_items:
        assert item["primary_stream"] == "Sanitary waste"
        assert any(
            "3(zp)" in ref.get("rule_or_section", "")
            for ref in item.get("source_references", [])
        )
