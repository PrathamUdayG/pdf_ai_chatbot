"""
PDF text extraction and document chunking.

Chunking Strategy
─────────────────
* RecursiveCharacterTextSplitter with 1 000-char chunks and 200-char overlap.
* Separators: paragraph → newline → sentence → word → char.
* Each chunk keeps metadata: source filename, page number, chunk index.
"""

from __future__ import annotations

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


# ── PDF Extraction ────────────────────────────────────────────────

def extract_documents(uploaded_files: list) -> list[Document]:
    """Read every page of every uploaded PDF and return LangChain Documents."""
    documents: list[Document] = []
    for pdf_file in uploaded_files:
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file.name,
                        "page": page_idx,
                        "total_pages": total_pages,
                    },
                )
            )
    return documents


# ── Chunking ──────────────────────────────────────────────────────

def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split page-level documents into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks
