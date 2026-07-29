import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME = "AI PDF Chat"

    API_VERSION = "1.0.0"

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    MODEL_NAME = os.getenv(
        "MODEL",
        "gemini-3.6-flash"
    )

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "gemini-embedding-2"
    )
    UPLOAD_DIRECTORY = "uploads"

    CHROMA_DIRECTORY = "chroma_db"


settings = Settings()