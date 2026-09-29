from pydantic import BaseModel, field_validator

MAX_QUESTION_LENGTH = 2000


class ChatRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()

        if not question:
            raise ValueError("The question must not be empty.")

        if len(question) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"The question must be at most {MAX_QUESTION_LENGTH} characters."
            )

        return question
