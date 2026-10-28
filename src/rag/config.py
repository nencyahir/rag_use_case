"""Paths and model names in one place."""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND / ".env")

PDF_DIR = Path(os.getenv("PDF_DIR", BACKEND.parent))
DATA_DIR = BACKEND / "data"
QDRANT_PATH = DATA_DIR / "qdrant"          # embedded Qdrant: no server or Docker needed
CACHE_DIR = DATA_DIR / "cache"             # vision descriptions, parsed structure
for d in (DATA_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

DOCS = {
    "gpo": PDF_DIR / "GPO-Anual-report.pdf",
    "gep": PDF_DIR / "GEP-Jun-2026.pdf",
    "imac": PDF_DIR / "iMac_PER_Oct2024.pdf",
}

DENSE_MODEL = "BAAI/bge-small-en-v1.5"      # 384 dims, local (ONNX, CPU)
DENSE_DIM = 384
SPARSE_MODEL = "Qdrant/bm25"                # keyword vectors, local
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # cross-encoder, local
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# GEP regional sections: canonical name -> aliases used to spot them in questions
REGIONS = {
    "East Asia and Pacific": ["east asia and pacific", "east asia", "eap"],
    "Europe and Central Asia": ["europe and central asia", "eca"],
    "Latin America and the Caribbean": ["latin america", "caribbean", "lac"],
    "Middle East, North Africa, Afghanistan, and Pakistan": ["middle east", "north africa", "mena", "mna"],
    "South Asia": ["south asia", "sar"],
    "Sub-Saharan Africa": ["sub-saharan africa", "sub saharan africa", "ssa"],
}
