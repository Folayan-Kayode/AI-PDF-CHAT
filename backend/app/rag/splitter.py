"""Page text chunking."""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    """Splits page text into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def split_pages(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return chunks tagged with their source page and chunk number."""
        chunks: list[dict[str, Any]] = []

        for page in pages:
            split = self.splitter.split_text(page["text"])

            for index, chunk in enumerate(split):
                text = chunk.strip()

                if not text:
                    continue

                chunks.append(
                    {
                        "text": text,
                        "page": page["page"],
                        "chunk": index + 1,
                    }
                )

        return chunks
