"""Chat orchestration."""

from typing import Any

from app.rag.pipeline import RAGPipeline, get_pipeline


class ChatService:
    """Answers questions using the shared RAG pipeline."""

    @staticmethod
    def chat(
        question: str,
        pipeline: RAGPipeline | None = None,
    ) -> dict[str, Any]:
        """Return {"answer": str, "sources": [...]} for the question."""
        return (pipeline or get_pipeline()).ask(question)
