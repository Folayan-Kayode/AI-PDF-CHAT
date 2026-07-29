from fastapi import APIRouter

from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post("/")
async def chat(
    request: ChatRequest
):

    answer = ChatService.chat(
    request.question
)

    return answer