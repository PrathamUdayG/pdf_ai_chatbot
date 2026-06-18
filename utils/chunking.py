from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = []
    for doc in documents:
        text_chunks = splitter.split_text(doc["text"])
        for chunk in text_chunks:
            chunks.append(
                {
                    "text": chunk,
                    "page": doc["page"],
                    "source": doc["source"]
                }
            )
    return chunks