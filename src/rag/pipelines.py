"""One pipeline per use case. Each can be built once (`build`) and then asked (`answer`).
Every answer carries a trace of the steps taken, so retrieval can be checked without an LLM."""
import json

from . import config
from .ingest.chunking import build_hierarchy, element_chunks, naive_chunks_with_pages, sentence_chunks, with_heading
from .retrieval.generate import ask
from .infra import vectordb
from .retrieval.postprocess import rerank, to_parents
from .retrieval.query import plan

PARENTS_FILE = lambda: config.CACHE_DIR / "gep_parents.json"


def collection(use_case: str, mode: str) -> str:
    return f"{use_case}_{mode}"


def build(use_case: str, mode: str = "advanced"):
    """Ingest the PDF for a use case and (re)create its Qdrant collection."""
    path = config.DOCS[use_case]
    name = collection(use_case, mode)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if use_case == "gpo":
        from .ingest.layout import read_pages
        pages = read_pages(path)
        chunks = naive_chunks_with_pages(pages) if mode == "naive" else sentence_chunks(pages)
        vectordb.build(name, chunks)
    elif use_case == "gep":
        from .ingest.structure import parse_report
        children, parents = build_hierarchy(parse_report(path))
        PARENTS_FILE().write_text(json.dumps(parents))
        vectordb.build(name, children, [with_heading(c) for c in children])
    elif use_case == "imac":
        chunks = element_chunks(path)
        if mode == "advanced":
            from .ingest.visuals import vision_chunks
            try:
                chunks += vision_chunks(path, start_id=len(chunks))
            except Exception as e:                      # no Gemini key: tables, text and footnotes still work
                print(f"vision skipped: {e}")
        vectordb.build(name, chunks)
    else:
        raise ValueError(use_case)
    return vectordb.client().count(name).count


def retrieve(use_case: str, question: str, mode: str = "advanced", use_llm: bool = True):
    """-> (hits, trace). No generation here."""
    name = collection(use_case, mode)
    trace = []
    if use_case == "gpo":
        if mode == "naive":
            hits = vectordb.dense_search(name, question, k=5)
            trace.append("dense top 5")
        else:
            hits = rerank(question, vectordb.hybrid_search(name, question, k=20), k=5)
            trace.append("hybrid (dense+BM25, RRF) top 20 -> cross-encoder rerank -> top 5")
    elif use_case == "gep":
        if mode == "naive":
            hits = vectordb.dense_search(name, question, k=5)
            trace.append("dense top 5")
        else:
            where, tags, query, how = plan(question, use_llm)
            trace.append(f"query understanding ({how}): filters={tags}, search query={query!r}")
            pool = vectordb.hybrid_search(name, query, k=20, where=where)
            if not pool and where is not None:
                pool = vectordb.hybrid_search(name, query, k=20)
                trace.append("filter matched nothing -> retried without filter")
            parents = {int(k): v for k, v in json.loads(PARENTS_FILE().read_text()).items()}
            hits = to_parents(rerank(question, pool, k=8), parents, n=3)
            trace.append("hybrid top 20 -> rerank -> parent sections (up to 3)")
    else:
        if mode == "naive":
            hits = vectordb.dense_search(name, question, k=5)
            trace.append("dense top 5")
        else:
            hits = vectordb.hybrid_search(name, question, k=6)
            trace.append("hybrid (dense+BM25, RRF) top 6 over table/text/footnote/vision chunks")
    return hits, trace


def answer(use_case: str, question: str, mode: str = "advanced", use_llm: bool = True) -> dict:
    hits, trace = retrieve(use_case, question, mode, use_llm)
    text = ask(question, hits)
    return {"question": question, "answer": text, "trace": trace,
            "citations": [{"n": i, "pages": f"{h['page_start']}-{h['page_end']}", "kind": h.get("kind"),
                           "heading": h.get("heading_path"), "text": h["text"][:300]} for i, h in enumerate(hits, 1)]}
