from app.rag.pipeline import RAGPipeline


class ChatService:

    @staticmethod
    def chat(question: str):

        pipeline = RAGPipeline()

        return pipeline.ask(question)