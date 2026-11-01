from pydantic import BaseModel, field_validator

MAX_QUESTION_LENGTH = 2000

MAX_DOCUMENT_ID_LENGTH = 128


class ChatRequest(BaseModel):
    question: str

    # Optional for backwards compatibility. When omitted, the newest document
    # is used; the UI always sends the document the user selected.
    document_id: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()

        if not question:
            raise ValueError("The question must not be empty.")

        if len(question) > MAX_QUESTION_LENGTH:
            raise ValueError(f"The question must be at most {MAX_QUESTION_LENGTH} characters.")

        return question

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        document_id = value.strip()

        if not document_id:
            return None

        if len(document_id) > MAX_DOCUMENT_ID_LENGTH:
            raise ValueError(
                f"The document id must be at most {MAX_DOCUMENT_ID_LENGTH} characters."
            )

        if "/" in document_id or "\\" in document_id:
            raise ValueError("The document id must not contain path separators.")

        return document_id
