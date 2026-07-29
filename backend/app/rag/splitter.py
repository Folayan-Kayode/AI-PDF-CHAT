from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
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

    def split_pages(self, pages):

        chunks = []

        for page in pages:

            split = self.splitter.split_text(page["text"])

            for i, chunk in enumerate(split):

                chunks.append(
                    {
                        "text": chunk,
                        "page": page["page"],
                        "chunk": i + 1,
                    }
                )

        return chunks