"""Chat endpoint."""

import logging
from typing import Any

from fastapi import APIRouter

from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# A plain def: retrieval and generation are blocking calls, so FastAPI runs
# this in its threadpool instead of blocking the event loop.
@router.post("/")
def chat(request: ChatRequest) -> dict[str, Any]:
    """Answer a question using the indexed document."""
    answer = ChatService.chat(request.question)

    logger.info(
        "answered question_chars=%s sources=%s",
        len(request.question),
        len(answer.get("sources") or []),
    )

    return answer
