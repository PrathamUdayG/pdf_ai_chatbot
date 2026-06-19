"""
LangChain LLM chains for question contextualisation and RAG answer generation.

Prompt Design
─────────────
1. **Contextualise chain** – rewrites follow-up questions into standalone
   queries using the recent chat history (last 6 messages).
2. **Answer chain** – grounded QA prompt that forces the LLM to answer
   ONLY from the retrieved context and cite sources.

Both chains use LCEL (LangChain Expression Language) for composability.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from config import LLM_MODEL


# ── LLM ───────────────────────────────────────────────────────────

def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """Return a streaming-capable Gemini model instance."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0.3,
        streaming=True,
    )


# ── Contextualise follow-up questions ─────────────────────────────

_CONTEXTUALISE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, "
            "rewrite the question as a standalone query that can be "
            "understood without the chat history. Do NOT answer it; "
            "just reformulate if needed, otherwise return as-is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def build_contextualise_chain(llm: ChatGoogleGenerativeAI):
    """LCEL chain: chat_history + input → standalone question string."""
    return _CONTEXTUALISE_PROMPT | llm | StrOutputParser()


# ── Answer generation ─────────────────────────────────────────────

_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful PDF assistant. Answer the user's question "
            "based ONLY on the provided context from uploaded PDF documents.\n\n"
            "Rules:\n"
            "1. Answer ONLY from the provided context.\n"
            "2. If the answer is not in the context, reply: "
            '"I could not find the answer in the uploaded PDFs."\n'
            "3. Cite the source document name and page number for every claim.\n"
            "4. Use clear Markdown formatting.\n"
            "5. Be thorough but concise.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def build_answer_chain(llm: ChatGoogleGenerativeAI):
    """LCEL chain: context + chat_history + input → streamed answer."""
    return _QA_PROMPT | llm | StrOutputParser()


# ── Helpers ───────────────────────────────────────────────────────

def format_documents(docs: list[Document]) -> str:
    """Render retrieved documents into a single context string."""
    parts: list[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {src} | Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def extract_citations(docs: list[Document]) -> list[dict]:
    """Pull citation metadata + excerpt from each retrieved document."""
    citations: list[dict] = []
    seen: set[tuple] = set()
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "excerpt": doc.page_content[:300],
            }
        )
    return citations


def history_to_langchain(messages: list[dict]) -> list:
    """Convert Streamlit session messages to LangChain message objects."""
    lc_messages = []
    for msg in messages[-6:]:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages
