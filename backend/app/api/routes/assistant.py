"""Conversational AI Assistant route contract."""

from fastapi import APIRouter, status
from ...core.config import settings
from ...schemas.assistant import AssistantChatRequest, AssistantChatResponse

router = APIRouter(prefix="/assistant", tags=["Conversational Assistant"])


@router.post(
    "/chat",
    response_model=AssistantChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Citizen Query Conversational Assistant (Contract)",
)
def assistant_chat(request: AssistantChatRequest):
    """API contract endpoint for citizen waste segregation conversational assistance.

    Currently reports a clean, authentic placeholder indicating that the Llama 3.2 Instruct
    ExecuTorch edge runtime is scheduled for subsequent integration.
    DOES NOT GENERATE FABRICATED AI RESPONSES.
    """
    return AssistantChatResponse(
        status="pending_model_integration",
        message=(
            f"Query received: '{request.message}'. "
            "Llama 3.2 Instruct conversational inference via ExecuTorch runtime on "
            f"{settings.target_hardware} is currently in contract stage and awaiting model deployment."
        ),
        target_hardware=settings.target_hardware,
        target_model="Llama 3.2 Instruct (1B/3B)",
        runtime="ExecuTorch Portable / NPU Delegate",
        language=request.language,
        is_integrated=False,
        response=None,
    )
