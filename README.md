<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LangChain-Text_Splitters-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Cost-$0_Free_Tier-brightgreen?style=for-the-badge" />
</p>

<h1 align="center">📄 PDF AI ChatBot</h1>

<p align="center">
  <b>An intelligent RAG-powered chatbot that lets you upload PDFs and have multi-turn conversations about them — powered by Google Gemini 2.5 Flash.</b>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-live-demo">Live Demo</a> •
  <a href="#️-architecture">Architecture</a> •
  <a href="#-design-decisions">Design Decisions</a> •
  <a href="#-setup-instructions">Setup</a> •
  <a href="#️-deployment">Deployment</a>
</p>

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 📤 **Multi-PDF Upload** | Upload and process multiple PDF files simultaneously |
| 🧠 **Conversational Memory** | Maintains chat history for context-aware follow-up questions |
| 🔍 **Semantic Search** | Uses `all-MiniLM-L6-v2` embeddings for accurate chunk retrieval |
| 💬 **Question Rewriting** | Automatically rewrites follow-ups like "explain that" into standalone queries |
| 📑 **Source Citations** | Shows exact page & source for every answer with expandable excerpts |
| ⚡ **Gemini 2.5 Flash** | Powered by Google's latest fast & efficient LLM (free tier) |
| 🗃️ **Persistent Vector Store** | ChromaDB stores embeddings locally for fast retrieval |

---

## 🔴 Live Demo

> **Deployed Application**: [https://prathamudayg-pdfaichatbot.streamlit.app](https://prathamudayg-pdfaichatbot.streamlit.app)
>
> *(After Streamlit Cloud deployment — see [Deployment](#️-deployment) section)*

---

## 🏗️ Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────┐
│                     STREAMLIT UI                         │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  Upload   │  │  Chat Input  │  │  Sidebar Status    │ │
│  │  PDFs     │  │  & Display   │  │  & File List       │ │
│  └────┬─────┘  └──────┬───────┘  └────────────────────┘ │
└───────┼───────────────┼──────────────────────────────────┘
        │               │
        ▼               ▼
┌───────────────┐ ┌─────────────────────────────────────┐
│  PDF Reader   │ │         LLM Pipeline                │
│  (pypdf)      │ │                                     │
│               │ │  1. rewrite_question()              │
│  Extract text │ │     └─ Gemini rewrites follow-ups   │
│  per page     │ │                                     │
└───────┬───────┘ │  2. retrieve_chunks()               │
        │         │     └─ ChromaDB semantic search      │
        ▼         │                                     │
┌───────────────┐ │  3. generate_answer()               │
│  Chunking     │ │     └─ Gemini generates response    │
│  (LangChain)  │ │     └─ Returns citations            │
│               │ └─────────────────────────────────────┘
│  800 chars    │
│  150 overlap  │
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────┐
│        ChromaDB Vector Store      │
│  ┌─────────────────────────────┐  │
│  │  SentenceTransformer        │  │
│  │  all-MiniLM-L6-v2          │  │
│  │  Encodes chunks → vectors  │  │
│  └─────────────────────────────┘  │
│  ┌─────────────────────────────┐  │
│  │  Persistent Collection      │  │
│  │  IDs + Embeddings + Metadata│  │
│  └─────────────────────────────┘  │
└───────────────────────────────────┘
```

### RAG Pipeline Flow

```
User Question
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Rewrite    │────▶│  Retrieve    │────▶│  Generate    │
│  Question   │     │  Top-K Chunks│     │  Answer      │
│  (Gemini)   │     │  (ChromaDB)  │     │  (Gemini)    │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │  Answer +    │
                                         │  Citations   │
                                         └──────────────┘
```

### Data Flow Summary

```
PDF Upload → pypdf (extract) → LangChain (chunk 800/150)
           → SentenceTransformer (embed) → ChromaDB (store)

Question → Gemini (rewrite) → ChromaDB (retrieve top-3)
         → Gemini (generate) → Answer + Citations → UI
```

---

## 🧠 Design Decisions

### 1. **Why RAG over Fine-Tuning?**
RAG (Retrieval-Augmented Generation) was chosen because it works with *any* PDF at runtime — no retraining needed. Users upload their own documents, so the system must handle arbitrary content dynamically.

### 2. **Why Gemini 2.5 Flash?**
- **Free tier** — zero cost for the entire project
- **Fast inference** — sub-second response times
- **Large context window** — handles complex multi-turn prompts easily
- No OpenAI API key or billing required

### 3. **Why ChromaDB (not FAISS or Pinecone)?**
- **Persistent storage** — survives app restarts without re-embedding
- **Zero infrastructure** — runs embedded, no external database server
- **Built-in metadata filtering** — stores page numbers and source filenames alongside embeddings
- Free and open-source

### 4. **Why `all-MiniLM-L6-v2` Embeddings?**
- **384-dimensional** — compact and fast
- **Top performer** on semantic textual similarity benchmarks for its size
- Runs locally — no external API calls for embedding, reducing latency

### 5. **Question Rewriting for Multi-Turn**
Instead of dumping the entire chat history into every retrieval query, the system uses Gemini to rewrite follow-ups like *"explain that more simply"* into standalone queries like *"Explain the transformer attention mechanism in simple terms"*. This dramatically improves retrieval accuracy.

### 6. **Chunking Strategy: 800 chars / 150 overlap**
- **800 characters** — small enough for precise retrieval, large enough to retain context
- **150-char overlap** — prevents information loss at chunk boundaries
- **`RecursiveCharacterTextSplitter`** — respects natural text boundaries (paragraphs → sentences → words)

### 7. **Streamlit for Frontend**
- **Rapid prototyping** — full chat UI in ~160 lines
- **Built-in chat components** — `st.chat_input`, `st.chat_message`, `st.expander`
- **Free cloud hosting** via Streamlit Cloud
- Session state manages conversation history seamlessly

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Web UI & chat interface |
| **LLM** | Google Gemini 2.5 Flash | Question rewriting & answer generation |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | Text → vector conversion |
| **Vector Store** | ChromaDB (Persistent) | Semantic similarity search |
| **Text Splitter** | LangChain `RecursiveCharacterTextSplitter` | Intelligent document chunking |
| **PDF Parser** | pypdf | PDF text extraction |

---

## 📁 Project Structure

```
pdf_ai_chatbot/
│
├── app.py                    # Main Streamlit application (entry point)
├── requirements.txt          # Python dependencies (7 packages)
├── .env                      # API keys (local only — gitignored)
├── .gitignore                # Git ignore rules
├── README.md                 # Documentation (this file)
│
├── .streamlit/
│   └── config.toml           # Streamlit dark theme configuration
│
├── utils/
│   ├── __init__.py           # Package initializer
│   ├── pdf_reader.py         # PDF text extraction (pypdf)
│   ├── chunking.py           # Text chunking (LangChain RecursiveCharacterTextSplitter)
│   ├── vector_store.py       # ChromaDB operations & SentenceTransformer embeddings
│   └── llm.py                # Gemini LLM — rewrite_question() + generate_answer()
│
└── chroma_db/                # ChromaDB persistent storage (gitignored)
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10** or higher
- A free [Google AI API Key](https://aistudio.google.com/apikey)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/PrathamUdayG/pdf_ai_chatbot.git
cd pdf_ai_chatbot

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file with your API key
echo GOOGLE_API_KEY=your_api_key_here > .env

# 5. Run the application
streamlit run app.py
```

The app will open at **http://localhost:8501**

### Usage

1. **Upload** one or more PDF files
2. Click **"Process PDFs"** — extracts text, creates chunks, stores embeddings
3. **Ask questions** in the chat box
4. View **source citations** with page numbers under each response
5. Ask **follow-up questions** — the bot understands conversation context

---

## ☁️ Deployment

### Deploy to Streamlit Cloud (Free)

1. **Push to GitHub** *(already done if you cloned this repo)*

2. **Go to [share.streamlit.io](https://share.streamlit.io)**
   - Click **"New app"**
   - Select repo: `PrathamUdayG/pdf_ai_chatbot`
   - Branch: `main`
   - Main file path: `app.py`

3. **Add Secrets** (Settings → Secrets):
   ```toml
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```

4. Click **Deploy!** 🎉

---

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI Studio API key for Gemini 2.5 Flash | ✅ Yes |

| Environment | How to Set |
|-------------|-----------|
| **Local** | `.env` file (gitignored) |
| **Streamlit Cloud** | Settings → Secrets |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/PrathamUdayG">Pratham Uday G</a>
</p>
