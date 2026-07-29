from pathlib import Path

from pypdf import PdfReader

from app.utils.helpers import clean_text

class PDFLoader:

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

    def load(self):

        reader = PdfReader(self.pdf_path)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = clean_text(page.extract_text() or "")

            pages.append(
                {
                    "page": page_number,
                    "text": text if text else ""
                }
            )

        return pages