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

    MODEL_NAME = _env("MODEL", "gemini-3.6-flash")

    EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "gemini-embedding-2")

    UPLOAD_DIRECTORY = "uploads"

    CHROMA_DIRECTORY = "chroma_db"

    MAX_UPLOAD_SIZE_MB = int(_env("MAX_UPLOAD_SIZE_MB", "25"))

    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def __init__(self):
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        if not self.GOOGLE_API_KEY or not self.GOOGLE_API_KEY.strip():
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. "
                "Copy .env.example to .env and add your Google API key."
            )

        self.GOOGLE_API_KEY = self.GOOGLE_API_KEY.strip()


settings = Settings()
