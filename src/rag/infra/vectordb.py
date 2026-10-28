"""Qdrant collection with two vectors per chunk (dense + BM25) and a metadata payload."""
from qdrant_client import QdrantClient, models

from .. import config
from . import embeddings as emb

_client = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(config.QDRANT_PATH))
    return _client


def exists(name: str) -> bool:
    return client().collection_exists(name)


def build(name: str, chunks: list[dict], embed_texts: list[str] | None = None, sparse: bool = True):
    """Create `name` fresh. `embed_texts[i]` is what gets embedded for chunks[i] (defaults to its text);
    the payload always keeps the original text."""
    c = client()
    if c.collection_exists(name):
        c.delete_collection(name)
    c.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=config.DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    texts = embed_texts or [ch["text"] for ch in chunks]
    dense = emb.dense_passages(texts)
    bm25 = emb.sparse_passages(texts)
    points = []
    for ch, d, s in zip(chunks, dense, bm25):
        points.append(models.PointStruct(
            id=ch["id"],
            vector={"dense": d, "bm25": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())},
            payload=ch))
    for i in range(0, len(points), 64):
        c.upsert(name, points[i:i + 64])


def dense_search(name: str, question: str, k: int = 5, where=None) -> list[dict]:
    res = client().query_points(name, query=emb.dense_query(question), using="dense", limit=k,
                                query_filter=where, with_payload=True)
    return [p.payload for p in res.points]


def hybrid_search(name: str, question: str, k: int = 5, where=None) -> list[dict]:
    """Dense + BM25, merged with Reciprocal Rank Fusion (ranks only, so no score scaling needed)."""
    s = emb.sparse_query(question)
    pool = max(k * 4, 20)
    res = client().query_points(
        name,
        prefetch=[
            models.Prefetch(query=emb.dense_query(question), using="dense", limit=pool, filter=where),
            models.Prefetch(query=models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                            using="bm25", limit=pool, filter=where)],
        query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k, with_payload=True)
    return [p.payload for p in res.points]
