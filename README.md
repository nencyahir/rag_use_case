# RAG Backend (3 use cases)

Retrieval-augmented question answering over three PDFs, each with its own ingestion and retrieval strategy. Every use case has a `naive` mode (dense top 5) and an `advanced` mode.

| Use case | Key | Pipeline |
|---|---|---|
| 1. GPO annual report | `gpo` | two-column reading order, PDF text repair, sentence chunks -> hybrid (dense + BM25, RRF) -> rerank |
| 2. World Bank GEP | `gep` | outline from fonts (chapter/region/box), parent/child chunks, heading-path embedding, LLM query routing + metadata filter, rerank, parent expansion |
| 3. Apple iMac | `imac` | rebuilt tables, footnote markers linked to endnotes, one chunk per element, Gemini page descriptions for charts |

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| PDF parsing | PyMuPDF |
| Embeddings | fastembed (local ONNX, CPU): dense `BAAI/bge-small-en-v1.5`, sparse `Qdrant/bm25` |
| Reranker | fastembed cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` |
| Vector DB | Qdrant, embedded local mode (no server or Docker) |
| Answer LLM / query routing | Groq (`openai/gpt-oss-120b` by default) |
| Vision (charts/pages) | Google Gemini (`gemini-3.5-flash-lite` by default) |
| API | FastAPI + Uvicorn, with a static frontend |
| Config | python-dotenv, Pydantic |

## Libraries (`requirements.txt`)

| Package | Used for |
|---|---|
| `pymupdf` | text, font and table extraction from PDFs |
| `qdrant-client` | embedded vector store, hybrid search, metadata filters |
| `fastembed` | dense/sparse embeddings and cross-encoder reranking |
| `fastapi`, `uvicorn` | REST API and web server |
| `pydantic` | request validation |
| `python-dotenv` | loading `.env` |
| `groq` | LLM answers and query routing |
| `google-genai` | Gemini vision descriptions of charts/pages |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env     # set GROQ_API_KEY and GEMINI_API_KEY
export PYTHONPATH=src
```

The source PDFs are not stored in this repo. Download them from the
[sample documents folder](https://drive.google.com/drive/folders/1Us5-u048KJZEkSTAqzJpFBCW05f8cRoH)
and place them in the parent folder (override with `PDF_DIR` in `.env`):
`GPO-Anual-report.pdf`, `GEP-Jun-2026.pdf`, `iMac_PER_Oct2024.pdf`.

## Usage

```bash
python -m rag.cli build all                                   # build indexes (re-run after adding GEMINI_API_KEY)
python -m rag.cli ask gep "What is the growth outlook for South Asia?"
python -m rag.eval                                            # retrieval check, no keys needed
python -m rag.eval --generate --llm                           # include answers
uvicorn rag.api:app                                           # UI at http://127.0.0.1:8000
```

API: `POST /{gpo|gep|imac}/ask` and `/retrieve` with `{"question": "...", "mode": "advanced"}`.
