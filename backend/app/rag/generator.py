from openai import OpenAI

from app.core.config import settings


class DeepSeekGenerator:
    """
    Generation via DeepSeek's OpenAI-compatible chat completions API.

    Embeddings remain on Google; only the generator is switched here.
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content
