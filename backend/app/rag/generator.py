from google import genai

from app.core.config import settings


class GeminiGenerator:

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.GOOGLE_API_KEY
        )

    def generate(self, prompt: str):

        response = self.client.models.generate_content(
            model=settings.MODEL_NAME,
            contents=prompt,
        )

        return response.text