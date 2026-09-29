import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    """
    Read an environment variable, treating a blank value as unset.

    Without this, a line like `MODEL=` in .env would yield "" instead of
    the default, since os.getenv only falls back when the variable is absent.
    """
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return value.strip()


class Settings:
    PROJECT_NAME = "AI PDF Chat"

    API_VERSION = "1.0.0"

    # Generation: DeepSeek (OpenAI-compatible API)
    MODEL_NAME = _env("MODEL", "deepseek-chat")

    DEEPSEEK_BASE_URL = _env(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    )

    # Embeddings: Google
    EMBEDDING_MODEL = _env(
        "EMBEDDING_MODEL",
        "gemini-embedding-2",
    )

    UPLOAD_DIRECTORY = "uploads"

    CHROMA_DIRECTORY = "chroma_db"

    MAX_UPLOAD_SIZE_MB = int(_env("MAX_UPLOAD_SIZE_MB", "25"))

    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def __init__(self):
        self.GOOGLE_API_KEY = self._require("GOOGLE_API_KEY")

        self.DEEPSEEK_API_KEY = self._require("DEEPSEEK_API_KEY")

    @staticmethod
    def _require(name: str) -> str:
        value = os.getenv(name)

        if not value or not value.strip():
            raise RuntimeError(
                f"{name} is not set. "
                f"Copy .env.example to .env and add your {name}."
            )

        return value.strip()


settings = Settings()
