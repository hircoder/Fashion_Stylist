import numpy as np
import pytest

from stylist.config import Settings
from stylist.embeddings import HashEmbedder, SentenceTransformerEmbedder, make_embedder


def test_hash_embedder_is_deterministic_and_normalized():
    e = HashEmbedder(dim=128)
    a = e.encode_docs(["red summer dress", "red summer dress"])
    assert a.shape == (2, 128) and a.dtype == np.float32
    assert np.allclose(a[0], a[1])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_hash_embedder_puts_similar_texts_closer():
    e = HashEmbedder(dim=256)
    docs = e.encode_docs(["womens beach sandals", "mens snow boots waterproof"])
    q = e.encode_queries(["beach sandals"])
    sims = docs @ q.T
    assert sims[0, 0] > sims[1, 0]


def test_hash_embedder_handles_empty_text():
    e = HashEmbedder(dim=64)
    v = e.encode_docs([""])
    assert v.shape == (1, 64)
    assert np.isfinite(v).all()


def test_make_embedder_returns_hash_embedder_when_configured():
    s = Settings.from_env({"EMBEDDER": "hash"})
    e = make_embedder(s)
    assert isinstance(e, HashEmbedder)
    assert e.name == "hash"


@pytest.mark.slow
def test_sentence_transformer_embedder_encodes_with_query_prefix():
    e = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", max_seq_length=256)
    docs = e.encode_docs(["Women's flat sandals for summer", "Men's insulated snow boots"])
    q = e.encode_queries(["sandals for the beach"])
    assert docs.shape == (2, 384) and e.dim == 384
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-3)
    sims = docs @ q.T
    assert sims[0, 0] > sims[1, 0]
    assert e.query_prefix.startswith("Represent this sentence")
