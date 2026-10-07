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

    # Retrieval telemetry is logged as structured fields on every request, so a
    # log query can answer "how often is the context empty" - the exact signal
    # that would have caught the retrieval regression. ``empty_context=True``
    # is the one to watch; it means the answer may rest on the profile alone.
    retrieval = answer.get("retrieval") or {}

    retrieved_chunks = retrieval.get("retrieved_chunks", 0)

    logger.info(
        "answered question_chars=%s sources=%s retrieved_chunks=%s "
        "considered_candidates=%s best_distance=%s profile_used=%s "
        "empty_context=%s",
        len(request.question),
        len(answer.get("sources") or []),
        retrieved_chunks,
        retrieval.get("considered_candidates"),
        retrieval.get("best_distance"),
        retrieval.get("profile_used"),
        retrieved_chunks == 0,
    )

    return answer
