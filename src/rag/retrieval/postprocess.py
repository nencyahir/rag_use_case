"""After retrieval: rerank a shortlist, or swap small matched chunks for their larger parent section."""
from ..infra.embeddings import cross_encoder


def rerank(question: str, hits: list[dict], k: int = 5) -> list[dict]:
    """A cross-encoder reads the question and each passage together and scores the pair.
    Slower than vector search, so it only runs on a shortlist."""
    if not hits:
        return []
    scores = list(cross_encoder().rerank(question, [h["text"] for h in hits]))
    ranked = sorted(zip(scores, hits), key=lambda pair: pair[0], reverse=True)
    return [h for _, h in ranked[:k]]


def to_parents(hits: list[dict], parents: dict, n: int = 3) -> list[dict]:
    """Swap each retrieved child for its parent window. Duplicates are dropped; rank order is kept."""
    out, seen = [], set()
    for h in hits:
        if h["parent_id"] in seen:
            continue
        seen.add(h["parent_id"])
        out.append({**h, **parents[h["parent_id"]]})
        if len(out) == n:
            break
    return out
