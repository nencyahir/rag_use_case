"""uvicorn rag.api:app --reload  (run from rag_backend with PYTHONPATH=src)"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config, pipelines

app = FastAPI(title="RAG use cases", docs_url=None, redoc_url=None, openapi_url=None)
FRONTEND = config.BACKEND / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND)


@app.get("/architecture", include_in_schema=False)
def architecture():
    return FileResponse(FRONTEND.with_name("architecture.html"))


class Ask(BaseModel):
    question: str
    mode: str = "advanced"          # "naive" or "advanced"


@app.get("/health")
def health():
    return {"ok": True, "use_cases": list(config.DOCS)}


@app.post("/{use_case}/ask")
def ask(use_case: str, body: Ask):
    if use_case not in config.DOCS or body.mode not in ("naive", "advanced"):
        raise HTTPException(404, "unknown use case or mode")
    return pipelines.answer(use_case, body.question, body.mode)


@app.post("/{use_case}/retrieve")
def retrieve(use_case: str, body: Ask):
    if use_case not in config.DOCS or body.mode not in ("naive", "advanced"):
        raise HTTPException(404, "unknown use case or mode")
    hits, trace = pipelines.retrieve(use_case, body.question, body.mode)
    return {"trace": trace, "hits": hits}
