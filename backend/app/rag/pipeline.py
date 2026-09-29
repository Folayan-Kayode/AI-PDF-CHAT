from app.rag.generator import GeminiGenerator
from app.rag.retriever import Retriever

NOT_FOUND_MESSAGE = (
    "I couldn't find that information in the uploaded document."
)


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.generator = GeminiGenerator()

    def ask(self, question: str):

        results = self.retriever.retrieve(question)

        documents = results.get("documents") or []
        metadata = results.get("metadata") or []

        if not documents:
            return {
                "answer": NOT_FOUND_MESSAGE,
                "sources": []
            }

        context = "\n\n".join(documents)

        prompt = f"""
You are a document question-answering assistant.

Rules:

1. Use ONLY the supplied context.
2. Do NOT use outside knowledge.
3. If the answer cannot be found in the context, say:
   "I couldn't find that information in the uploaded document."
4. Keep your answer concise and accurate.
5. Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = self.generator.generate(prompt)

        # Only return sources when the model actually answered the question.
        if NOT_FOUND_MESSAGE.lower() in (answer or "").lower():
            return {
                "answer": answer,
                "sources": []
            }

        return {
            "answer": answer,
            "sources": metadata
        }