"""Automated API Integration Tests for EcoSort AI FastAPI Backend.

Validates:
1. Health endpoint with knowledge-base status.
2. Waste categories overview (four statutory streams).
3. Waste item retrieval by class_id.
4. Unknown class_id 404 error handling.
5. Search endpoint (English and Tamil aliases).
6. Stream-level retrieval for statutory streams.
7. Invalid stream 400 error handling.
8. Statutory source references preservation.
9. Verification status preservation (verified vs pending).
10. Vision endpoint contract (placeholder without fake YOLO results).
11. Assistant endpoint contract (placeholder without fake LLM answers).
12. Languages endpoint (distinguishing implemented from planned).
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_STREAMS = {"Wet waste", "Dry waste", "Sanitary waste", "Special care waste"}


@pytest.fixture(scope="module")
def client():
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------
# 1. Health Endpoint Tests
# --------------------------------------------------------------------

def test_api_health_endpoint(client):
    """Verify GET /api/health returns app info and knowledge-base availability."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "EcoSort AI"
    assert "version" in data
    assert "target_hardware" in data
    assert "knowledge_base" in data

    kb = data["knowledge_base"]
    assert kb["status"] == "available"
    assert kb["total_items"] >= 40
    assert kb["verified_items"] >= 40
    assert "Solid Waste Management Rules, 2026" in kb["primary_legislation"]


def test_root_and_direct_health(client):
    """Verify root GET / and direct GET /health convenience routes."""
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert resp_root.json()["app"] == "EcoSort AI"

    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"


# --------------------------------------------------------------------
# 2. Waste Categories & Stream Tests
# --------------------------------------------------------------------

def test_get_waste_categories(client):
    """Verify GET /api/waste/categories returns the four statutory streams."""
    response = client.get("/api/waste/categories")
    assert response.status_code == 200
    categories = response.json()
    assert len(categories) == 4

    stream_names = {cat["stream_name"] for cat in categories}
    assert stream_names == VALID_STREAMS

    for cat in categories:
        assert cat["bin_color"] is not None
        assert cat["statutory_definition"] is not None
        assert cat["legal_citation"] is not None
        assert cat["item_count"] > 0


def test_get_items_by_valid_stream(client):
    """Verify GET /api/waste/stream/{stream_name} returns filtered items."""
    for stream in VALID_STREAMS:
        response = client.get(f"/api/waste/stream/{stream}")
        assert response.status_code == 200
        items = response.json()
        assert len(items) > 0
        for item in items:
            assert item["primary_stream"] == stream


def test_get_items_by_stream_with_normalization(client):
    """Verify stream name normalization (e.g. 'wet_waste' or 'dry-waste')."""
    resp1 = client.get("/api/waste/stream/wet_waste")
    assert resp1.status_code == 200
    assert len(resp1.json()) > 0

    resp2 = client.get("/api/waste/stream/sanitary-waste")
    assert resp2.status_code == 200
    assert len(resp2.json()) > 0


def test_invalid_stream_handling(client):
    """Verify GET /api/waste/stream/{stream_name} returns 400 on invalid stream."""
    response = client.get("/api/waste/stream/non_existent_stream")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Invalid waste stream" in detail
    assert "Solid Waste Management Rules, 2026" in detail or "SWM Rules 2026" in detail


# --------------------------------------------------------------------
# 3. Waste Item Retrieval & Error Handling Tests
# --------------------------------------------------------------------

def test_get_valid_waste_item(client):
    """Verify GET /api/waste/{class_id} returns complete item details."""
    response = client.get("/api/waste/dry_plastic_pet_bottles")
    assert response.status_code == 200
    item = response.json()
    assert item["class_id"] == "dry_plastic_pet_bottles"
    assert item["primary_stream"] == "Dry waste"
    assert "PET" in item["item_name"]
    assert item["verification_status"] == "verified"
    assert len(item["source_references"]) > 0


def test_unknown_waste_item_404(client):
    """Verify GET /api/waste/{class_id} returns 404 for unknown IDs."""
    response = client.get("/api/waste/unknown_item_class_12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_waste_item_recommendation(client):
    """Verify GET /api/waste/recommendation/{class_id} returns source-backed directives."""
    response = client.get("/api/waste/recommendation/sanitary_pads_towels")
    assert response.status_code == 200
    rec = response.json()
    assert rec["class_id"] == "sanitary_pads_towels"
    assert rec["primary_stream"] == "Sanitary waste"
    assert rec["is_pending_verification"] is False
    assert len(rec["source_references"]) > 0
    assert any("3(zp)" in ref["rule_or_section"] for ref in rec["source_references"])


# --------------------------------------------------------------------
# 4. Search Tests (English & Tamil)
# --------------------------------------------------------------------

def test_search_waste_items_english(client):
    """Verify GET /api/waste/search?q={query} with English terms."""
    response = client.get("/api/waste/search?q=bottle")
    assert response.status_code == 200
    items = response.json()
    assert len(items) >= 2
    class_ids = [i["class_id"] for i in items]
    assert "dry_plastic_pet_bottles" in class_ids


def test_search_waste_items_tamil_alias(client):
    """Verify GET /api/waste/search?q={query} with Tamil aliases."""
    response = client.get("/api/waste/search?q=பாட்டில்")  # Tamil for bottle
    assert response.status_code == 200
    items = response.json()
    assert len(items) >= 1
    assert any("bottle" in i["class_id"] for i in items)


def test_search_waste_items_empty_query(client):
    """Verify empty search query returns empty list or validates query."""
    response = client.get("/api/waste/search?q=%20")
    assert response.status_code == 200
    assert response.json() == []


# --------------------------------------------------------------------
# 5. Traceability & Verification Status Preservation Tests
# --------------------------------------------------------------------

def test_source_references_preserved_in_api(client):
    """Verify that source references contain authentic fields in API response."""
    response = client.get("/api/waste/special_cfl_tube_lights")
    assert response.status_code == 200
    item = response.json()
    refs = item["source_references"]
    assert len(refs) > 0
    ref = refs[0]
    assert "Solid Waste Management Rules, 2026" in ref["document_title"]
    assert "Ministry of Environment, Forest and Climate Change" in ref["issuing_organisation"]
    assert "3(zx)" in ref["rule_or_section"]
    assert ref["page_number"] is not None


def test_pending_verification_preservation(client):
    """Verify that pending verification entries retain pending status."""
    response = client.get("/api/waste/pending_cosmetic_aerosols")
    assert response.status_code == 200
    item = response.json()
    assert item["verification_status"] == "pending_verification"

    # Also check recommendation endpoint preserves pending flag
    rec_resp = client.get("/api/waste/recommendation/pending_cosmetic_aerosols")
    assert rec_resp.status_code == 200
    assert rec_resp.json()["is_pending_verification"] is True


# --------------------------------------------------------------------
# 6. Vision & Assistant Placeholder Contract Tests
# --------------------------------------------------------------------

def test_vision_detect_contract_placeholder(client):
    """Verify POST /api/vision/detect operates cleanly without fabricated YOLO results when weights pending."""
    from PIL import Image
    test_img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    test_img.save(buf, format="JPEG")
    valid_image_bytes = buf.getvalue()

    files = {"file": ("test_frame.jpg", io.BytesIO(valid_image_bytes), "image/jpeg")}
    data = {"language": "en"}

    response = client.post("/api/vision/detect", files=files, data=data)
    assert response.status_code == 200
    payload = response.json()

    # STRICT RULE: Must be transparently identified without hallucinating fake detections
    assert payload["status"] in ("model_weights_not_found", "success")
    assert payload["is_integrated"] is True
    assert "Raspberry Pi 5" in payload["target_hardware"]
    assert "yolo11n" in payload["target_model"].lower()


def test_assistant_chat_contract_placeholder(client):
    """Verify POST /api/assistant/chat returns placeholder without fabricated LLM responses."""
    body = {
        "message": "Where should I throw broken mercury thermometers?",
        "language": "en",
    }
    response = client.post("/api/assistant/chat", json=body)
    assert response.status_code == 200
    payload = response.json()

    # STRICT RULE: Must be clearly identified as pending/contract stage
    assert payload["status"] == "pending_model_integration"
    assert payload["is_integrated"] is False
    assert payload["response"] is None
    assert "Llama 3.2" in payload["target_model"]
    assert "ExecuTorch" in payload["runtime"]


# --------------------------------------------------------------------
# 7. Languages Endpoint Tests
# --------------------------------------------------------------------

def test_languages_endpoint(client):
    """Verify GET /api/languages distinguishes implemented from planned languages."""
    response = client.get("/api/languages")
    assert response.status_code == 200
    data = response.json()

    impl_codes = [lang["code"] for lang in data["implemented_languages"]]
    assert "en" in impl_codes
    assert "ta" in impl_codes

    plan_codes = [lang["code"] for lang in data["planned_languages"]]
    assert "hi" in plan_codes
    assert "te" in plan_codes
