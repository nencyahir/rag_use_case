"""Local embedding models: dense (meaning), sparse BM25 (exact words), and a cross-encoder reranker."""
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from .. import config


@lru_cache(maxsize=1)
def _dense():
    return TextEmbedding(config.DENSE_MODEL)


@lru_cache(maxsize=1)
def _sparse():
    return SparseTextEmbedding(config.SPARSE_MODEL)


@lru_cache(maxsize=1)
def cross_encoder():
    return TextCrossEncoder(config.RERANK_MODEL)


def dense_passages(texts):
    return [v.tolist() for v in _dense().embed(texts, batch_size=32)]


def dense_query(question):
    return next(iter(_dense().embed([config.BGE_QUERY_PREFIX + question]))).tolist()


def sparse_passages(texts):
    return list(_sparse().embed(texts))


def sparse_query(question):
    return next(iter(_sparse().query_embed(question)))
