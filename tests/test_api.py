import pytest
from fastapi.testclient import TestClient

from stylist.api import create_app
from stylist.config import Settings
from stylist.service import RecommendationService


@pytest.fixture(scope="module")
def client(fixture_index, hash_embedder):
    settings = Settings.from_env({"EMBEDDER": "hash"})
    svc = RecommendationService(fixture_index, hash_embedder, settings, llm=None)
    app = create_app(settings, service=svc)
    with TestClient(app) as c:
        yield c


def test_recommend_returns_contract_fields(client):
    r = client.post("/recommend", json={"query": "snow boots", "k": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {
        "request_id",
        "query",
        "plan",
        "slots",
        "note",
        "warnings",
        "index_info",
        "llm_info",
        "timings",
    }
    slot = body["slots"][0]
    assert slot["name"] and len(slot["items"]) <= 2
    item = slot["items"][0]
    for key in (
        "rank",
        "row_id",
        "parent_asin",
        "title",
        "price",
        "price_known",
        "url",
        "image_url",
        "average_rating",
        "rating_number",
        "reason",
        "matched_keywords",
    ):
        assert key in item
    assert body["llm_info"]["planner_used"] == "heuristic"


def test_recommend_rejects_blank_query_with_error_body(client):
    r = client.post("/recommend", json={"query": "   "})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    assert "blank" in r.json()["error"]["message"]


def test_recommend_rejects_inverted_price_window(client):
    r = client.post("/recommend", json={"query": "hat", "min_price": 50, "max_price": 10})
    assert r.status_code == 422
    assert "min_price" in r.json()["error"]["message"]


def test_recommend_rejects_bad_k(client):
    assert client.post("/recommend", json={"query": "hat", "k": 0}).status_code == 422
    assert client.post("/recommend", json={"query": "hat", "k": 11}).status_code == 422


def test_recommend_strict_price_filter_through_api(client):
    r = client.post("/recommend", json={"query": "swimsuit", "max_price": 5})
    assert r.status_code == 200
    for item in r.json()["slots"][0]["items"]:
        assert item["price_known"] and item["price"] <= 5


def test_health_is_always_ok_and_describes_state(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["index_loaded"] is True
    assert body["index"]["rows"] > 0
    assert body["llm"] == {"provider": None, "model": None}


def test_ready_is_200_when_index_loaded(client):
    assert client.get("/ready").status_code == 200


def test_ready_is_503_when_index_missing(tmp_path):
    settings = Settings.from_env({"EMBEDDER": "hash", "INDEX_DIR": str(tmp_path / "nope")})
    with TestClient(create_app(settings)) as c:
        r = c.get("/ready")
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "index_not_loaded"
        assert c.get("/health").status_code == 200
        assert c.post("/recommend", json={"query": "x"}).status_code == 503


def test_openapi_describes_recommend(client):
    spec = client.get("/openapi.json").json()
    assert "/recommend" in spec["paths"]
    post = spec["paths"]["/recommend"]["post"]
    assert "RecommendRequest" in str(post["requestBody"])
    assert spec["info"]["title"]


def test_root_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_retrieval_timeout_is_a_504_with_error_body(fixture_index, hash_embedder):
    import time

    settings = Settings.from_env({"EMBEDDER": "hash", "REQUEST_DEADLINE_S": "0.5"})
    svc = RecommendationService(fixture_index, hash_embedder, settings, llm=None)
    real = svc.retriever.retrieve

    def slow(*a, **kw):
        time.sleep(1.2)
        return real(*a, **kw)

    svc.retriever.retrieve = slow
    with TestClient(create_app(settings, service=svc)) as c:
        r = c.post("/recommend", json={"query": "boots"})
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "deadline_exceeded"


def test_unknown_route_uses_the_error_body_shape(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "http_404"


def test_load_error_is_not_leaked_verbatim(tmp_path):
    settings = Settings.from_env({"EMBEDDER": "hash", "INDEX_DIR": str(tmp_path / "nope")})
    with TestClient(create_app(settings)) as c:
        body = c.get("/ready").json()
        assert str(tmp_path) not in body["error"]["message"]
        assert body["error"]["message"]  # still says something useful
        health = c.get("/health").json()
        assert health["load_error"] and str(tmp_path) not in health["load_error"]


def test_embedder_dimension_mismatch_is_rejected_at_startup(tmp_path, fixture_catalog):
    from stylist.embeddings import HashEmbedder
    from stylist.index import build_index

    build_index(
        fixture_catalog, tmp_path / "idx", HashEmbedder(dim=64), limit=20, sampling="popular"
    )
    settings = Settings.from_env({"EMBEDDER": "hash", "INDEX_DIR": str(tmp_path / "idx")})
    with TestClient(create_app(settings)) as c:  # HashEmbedder() default dim is 256
        assert c.get("/ready").status_code == 503
        assert "dim" in c.get("/health").json()["load_error"]


def test_overview_deck_is_served(client):
    r = client.get("/overview")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "what i built" in r.text.lower()
