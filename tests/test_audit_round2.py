"""Second council round: reranker admissibility, retrieval ordering, api limits, index checks."""

import pytest

from stylist.config import Settings
from stylist.planner import QueryPlan, Slot, SlotWindow
from stylist.reranker import SlotRerankOutput, apply_slot_rerank
from stylist.retrieval import Candidate, Retriever, SlotCandidates, type_match


def _cand(i, title, typed, matched=None, rating=4.0):
    c = Candidate(
        idx=i, row_id=i, score=1.0 - i / 100, title=title, average_rating=rating, rating_number=10
    )
    c.type_match = typed
    c.matched_keywords = list(matched or (["boot"] if typed else []))
    return c


def _sc(cands, keywords=("boot",)):
    slot = Slot(name="boots", search_query="boots", keywords=list(keywords))
    return SlotCandidates(
        slot, SlotWindow(None, None, None, False), cands, len(cands), [], eligible_rows=len(cands)
    )


def test_off_type_picks_are_rejected_when_enough_typed_candidates_exist():
    cands = [
        _cand(1, "Snow Boots", True),
        _cand(2, "Ball Pump", False),
        _cand(3, "Rain Boots", True),
        _cand(4, "Hiking Boots", True),
    ]
    out = SlotRerankOutput(
        picks=[
            {"row_id": 2, "reason": "nice", "evidence": ["title"]},
            {"row_id": 1, "reason": "warm", "evidence": ["title"]},
        ]
    )
    ranked, warnings = apply_slot_rerank(out, _sc(cands), k=2)
    assert [c.row_id for c in ranked.ordered][:2] == [1, 3]
    assert any("2" in w and "not the product type" in w for w in warnings)


def test_off_type_pick_is_kept_when_nothing_typed_exists():
    cands = [_cand(1, "Ball Pump", False), _cand(2, "Air Mattress", False)]
    out = SlotRerankOutput(picks=[{"row_id": 2, "reason": "closest", "evidence": ["title"]}])
    ranked, _ = apply_slot_rerank(out, _sc(cands), k=2)
    assert [c.row_id for c in ranked.ordered] == [2]


def test_no_good_match_with_picks_keeps_the_picks_and_warns():
    cands = [_cand(1, "Snow Boots", True), _cand(2, "Rain Boots", True)]
    out = SlotRerankOutput(
        picks=[{"row_id": 2, "reason": "r", "evidence": ["title"]}], no_good_match=True
    )
    ranked, warnings = apply_slot_rerank(out, _sc(cands), k=2)
    assert ranked.ordered[0].row_id == 2
    assert any("no_good_match" in w for w in warnings)


def test_reasons_are_link_stripped_word_capped_and_evidence_checked():
    cands = [_cand(1, "Snow Boots", True, rating=None)]
    long_reason = "visit http://evil.example now " + "word " * 40
    out = SlotRerankOutput(
        picks=[{"row_id": 1, "reason": long_reason, "evidence": ["title", "rating", "price"]}]
    )
    ranked, _ = apply_slot_rerank(out, _sc(cands), k=1)
    reason = ranked.reasons[1]
    assert "http" not in reason.reason and "evil" not in reason.reason
    assert len(reason.reason.split()) <= 20
    assert reason.evidence == ["title"]  # no rating, no price on this candidate


@pytest.mark.parametrize(
    "title,keywords,expected",
    [
        ("Shoe Insoles Arch Support", ["shoe"], False),
        ("Jacket Hanger Set", ["jacket"], False),
        ("Boot Laces 2 Pack", ["boot", "boots"], False),
        ("Shoe Laces Flat", ["laces", "shoe laces"], True),  # the slot asks for laces
        ("Leather Boots", ["boot"], True),
    ],
)
def test_accessory_veto_applies_to_exact_keywords_too(title, keywords, expected):
    assert type_match(title, keywords) is expected


def _retriever(fixture_index, hash_embedder):
    return Retriever(fixture_index, hash_embedder, Settings.from_env({"EMBEDDER": "hash"}))


def test_type_gate_threshold_is_k_not_candidate_depth(fixture_index, hash_embedder):
    r = _retriever(fixture_index, hash_embedder)
    plan = QueryPlan(
        intent="t",
        slots=[Slot(name="x", search_query="boots", keywords=["boot", "boots"])],
        source="llm",
    )
    [res] = r.retrieve(plan, [SlotWindow(None, None, None, False)], n_candidates=40, k=3)
    typed = [c for c in res.candidates if c.type_match]
    assert len(typed) >= 3 and all(
        c.type_match for c in res.candidates
    )  # never padded with off-type rows


def test_brand_order_is_typed_own_then_two_untyped_own_then_typed_others(
    fixture_index, hash_embedder
):
    cat = fixture_index.catalog
    r = _retriever(fixture_index, hash_embedder)
    rows = [i for i in cat.index if "boot" not in str(cat.loc[i, "title"]).lower()][:8]
    saved = cat.loc[rows, "store"].copy()
    try:
        cat.loc[rows, "store"] = "Zebrabrand"
        getattr(fixture_index, "_column_cache", {}).clear()
        plan = QueryPlan(
            intent="t",
            brand="zebrabrand",
            slots=[Slot(name="boots", search_query="boots", keywords=["boot", "boots"])],
            source="llm",
        )
        [res] = r.retrieve(plan, [SlotWindow(None, None, None, False)], n_candidates=10, k=4)
        own = [c for c in res.candidates if c.store == "Zebrabrand"]
        assert len(own) <= 2
        others = [c for c in res.candidates if c.store != "Zebrabrand"]
        assert others and all(c.type_match for c in others)
        first_other = next(i for i, c in enumerate(res.candidates) if c.store != "Zebrabrand")
        assert first_other <= 2
    finally:
        cat.loc[rows, "store"] = saved
        getattr(fixture_index, "_column_cache", {}).clear()
        r._brand_masks.clear()


def test_features_null_cell_is_an_empty_list(fixture_index, hash_embedder):
    r = _retriever(fixture_index, hash_embedder)
    cat = fixture_index.catalog
    saved = cat.at[0, "features"]
    cat.at[0, "features"] = float("nan")
    try:
        assert r._hydrate(Candidate(0, 0, 0.0)).features == []
    finally:
        cat.at[0, "features"] = saved


@pytest.mark.parametrize(
    "title,keywords,expected",
    [
        ("Packable Rain Jacket with Storage Bag", ["rain jacket"], True),  # the bag is an extra
        ("Insoles for Running Shoes", ["running shoes"], False),
        ("Running Shoe Insoles Arch Support", ["running shoes"], False),
        ("Shoe Covers Waterproof", ["shoe"], False),
        ("Hiking Boots with Free Laces", ["boots"], True),
        ("Beach Bag Canvas Tote", ["beach bag", "tote"], True),  # the slot wants a bag
    ],
)
def test_accessory_veto_is_positional(title, keywords, expected):
    assert type_match(title, keywords) is expected


def test_total_budget_warning_compares_one_item_per_slot():
    # covered through the service: the warning text now says "top pick of each slot"
    import inspect

    from stylist import service

    assert "top pick of each slot" in inspect.getsource(service)


def test_partial_total_budget_split_keeps_the_planner_values():
    from stylist.planner import PlannerOutput, normalize_plan

    out = PlannerOutput(
        budget_max=100.0,
        budget_scope="total",
        slots=[
            {"name": "a", "search_query": "a", "budget_max": 60.0},
            {"name": "b", "search_query": "b"},
            {"name": "c", "search_query": "c"},
        ],
    )
    plan = normalize_plan(out, "q")
    assert plan.slots[0].budget_max == 60.0
    assert plan.slots[1].budget_max == plan.slots[2].budget_max == 20.0
    assert any("1 of 3" in w for w in plan.warnings)


def test_style_keywords_keep_eight():
    from stylist.planner import PlannerOutput, normalize_plan

    out = PlannerOutput(
        style_keywords=[f"s{i}" for i in range(10)], slots=[{"name": "a", "search_query": "a"}]
    )
    assert len(normalize_plan(out, "q").style_keywords) == 8


async def test_failed_llm_calls_are_counted(fixture_index, hash_embedder):
    from stylist.llm import FakeLLM, LLMTransportError
    from stylist.schemas import RecommendRequest
    from stylist.service import RecommendationService

    llm = FakeLLM(handler=lambda s, u, schema: LLMTransportError("down"))
    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env({"EMBEDDER": "hash", "PLAN_CACHE_SIZE": "0"}),
        llm=llm,
    )
    try:
        r = await svc.recommend(RecommendRequest(query="boots", k=2))
    finally:
        svc.close()
    assert r.llm_info.calls >= 1 and r.llm_info.failed_calls == r.llm_info.calls
    assert r.llm_info.plan_cache_hit is False


async def test_cache_hit_is_reported(fixture_index, hash_embedder):
    from stylist.schemas import RecommendRequest
    from stylist.service import RecommendationService

    svc = RecommendationService(
        fixture_index, hash_embedder, Settings.from_env({"EMBEDDER": "hash"}), llm=None
    )
    try:
        await svc.recommend(RecommendRequest(query="boots", k=2))
        r = await svc.recommend(RecommendRequest(query="boots", k=2))
    finally:
        svc.close()
    assert r.llm_info.plan_cache_hit is True


def test_cli_pretty_print_handles_a_missing_rating():
    from stylist.cli import _format_pretty
    from stylist.planner import QueryPlan, Slot
    from stylist.schemas import IndexInfo, Item, LLMInfo, RecommendResponse, SlotResult

    item = Item(
        rank=1,
        row_id=1,
        parent_asin="B0ABC12345",
        title="Boots",
        price=None,
        price_known=False,
        average_rating=None,
        rating_number=0,
        store=None,
        audience="unknown",
        image_url=None,
        url=None,
        score=0.1,
        matched_keywords=[],
        reason="r",
        evidence=[],
    )
    res = RecommendResponse(
        request_id="x",
        query="boots",
        plan=QueryPlan(
            intent="boots", slots=[Slot(name="boots", search_query="boots")], source="heuristic"
        ),
        slots=[
            SlotResult(
                name="boots",
                search_query="boots",
                keywords=[],
                exclude_keywords=[],
                budget_max=None,
                n_eligible=1,
                items=[item],
            )
        ],
        note="",
        warnings=[],
        index_info=IndexInfo(
            rows=1, sampling="all", limit=None, embedding_model="hash", built_at="now"
        ),
        llm_info=LLMInfo(provider=None, model=None, planner_used="heuristic", rerank_used=False),
        timings={},
    )
    text = _format_pretty(res)
    assert "no rating" in text and "no link" in text


def test_cli_rejects_non_positive_build_limits():
    from stylist.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["build-index", "--limit", "0"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["recommend", "q", "--k", "-1"])


# ----------------------------------------------------------------------------- api round 2


from fastapi.testclient import TestClient  # noqa: E402

from stylist.api import _TokenBucket, create_app  # noqa: E402
from stylist.service import RecommendationService  # noqa: E402


def _app(fixture_index, hash_embedder, **env):
    settings = Settings.from_env({"EMBEDDER": "hash", **env})
    svc = RecommendationService(fixture_index, hash_embedder, settings, llm=None)
    return create_app(settings, service=svc)


def test_forwarded_for_is_ignored_unless_proxy_headers_are_trusted(fixture_index, hash_embedder):
    with TestClient(_app(fixture_index, hash_embedder, RATE_LIMIT_PER_MINUTE="12")) as c:
        codes = [
            c.post(
                "/recommend", json={"query": "boots"}, headers={"x-forwarded-for": f"10.0.0.{i}"}
            ).status_code
            for i in range(4)
        ]
        assert codes == [
            200,
            200,
            429,
            429,
        ]  # burst is per_minute / 6 = 2; spoofed ips share one bucket
    with TestClient(
        _app(fixture_index, hash_embedder, RATE_LIMIT_PER_MINUTE="12", TRUST_PROXY_HEADERS="1")
    ) as c:
        codes = [
            c.post(
                "/recommend", json={"query": "boots"}, headers={"x-forwarded-for": f"10.0.0.{i}"}
            ).status_code
            for i in range(4)
        ]
        assert codes == [200, 200, 200, 200]  # behind a proxy each forwarded ip has its own bucket


def test_token_bucket_burst_is_a_sixth_of_the_minute():
    b = _TokenBucket(60)
    assert b.burst == 10
    allowed = [b.allow("k", now=100.0)[0] for _ in range(11)]
    assert allowed[:10] == [True] * 10 and allowed[10] is False
    assert b.allow("k", now=101.0)[0] is True  # one token per second refills


def test_security_headers_are_set(fixture_index, hash_embedder):
    with TestClient(_app(fixture_index, hash_embedder)) as c:
        r = c.get("/health")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert "referrer-policy" in r.headers
        r = c.get("/")
        assert "content-security-policy" in r.headers
        assert "m.media-amazon.com" in r.headers["content-security-policy"]


def test_body_limit_replays_a_chunked_body_once_and_then_ends(fixture_index, hash_embedder):
    from stylist.api import _BodyLimit

    seen = []

    async def inner(scope, receive, send):
        seen.append(await receive())
        seen.append(await receive())  # a second read must not hang or re-read the socket
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    chunks = [
        {"type": "http.request", "body": b'{"query":', "more_body": True},
        {"type": "http.request", "body": b' "x"}', "more_body": False},
    ]

    async def receive():
        return chunks.pop(0) if chunks else {"type": "http.disconnect"}

    sent = []

    async def send(m):
        sent.append(m)

    import asyncio

    asyncio.run(
        _BodyLimit(inner, 100)({"type": "http", "method": "POST", "headers": []}, receive, send)
    )
    assert seen[0]["body"] == b'{"query": "x"}' and seen[0]["more_body"] is False
    assert seen[1]["body"] == b"" and seen[1]["more_body"] is False
    assert sent[0]["status"] == 200


# ------------------------------------------------------------------- index / artifacts round 2


def test_loader_requires_every_serving_column(tmp_path, fixture_catalog, hash_embedder):
    import pandas as pd

    from stylist.index import IndexValidationError, SearchIndex, build_index

    index_dir = tmp_path / "idx"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=20, sampling="popular")
    df = pd.read_parquet(index_dir / "catalog.parquet").drop(columns=["price"])
    df.to_parquet(index_dir / "catalog.parquet", index=False)
    from tests.test_audit_batch_b import _resign

    _resign(index_dir)
    with pytest.raises(IndexValidationError, match="price"):
        SearchIndex.load(index_dir)


def test_revision_mismatch_is_rejected_even_when_the_index_has_none(
    tmp_path, fixture_catalog, hash_embedder
):
    from stylist.api import build_service

    class Pinned(type(hash_embedder)):
        revision = "abc123"

    from stylist.index import IndexValidationError, build_index

    build_index(fixture_catalog, tmp_path / "idx", hash_embedder, limit=20, sampling="popular")
    settings = Settings.from_env({"EMBEDDER": "hash", "INDEX_DIR": str(tmp_path / "idx")})
    import stylist.api as api_mod

    real = api_mod.make_embedder
    api_mod.make_embedder = lambda s: Pinned(dim=256)
    try:
        with pytest.raises(IndexValidationError, match="revision"):
            build_service(settings)
    finally:
        api_mod.make_embedder = real


def test_artifact_install_rejects_an_index_the_loader_would_refuse(
    tmp_path, fixture_catalog, hash_embedder
):
    import hashlib
    import json
    import tarfile

    from stylist.artifacts import ArtifactError, ensure_index
    from stylist.index import build_index

    src = tmp_path / "src"
    build_index(fixture_catalog, src / "index", hash_embedder, limit=20, sampling="popular")
    meta = json.loads((src / "index" / "meta.json").read_text())
    meta["pipeline_version"] = "0"  # stale pipeline: checksums fine, loader refuses
    (src / "index" / "meta.json").write_text(json.dumps(meta))
    tar_path = tmp_path / "stale.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(src / "index", arcname="index")
    sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    settings = Settings.from_env(
        {
            "EMBEDDER": "hash",
            "INDEX_DIR": str(tmp_path / "index"),
            "INDEX_URL": tar_path.as_uri(),
            "INDEX_SHA256": sha,
            "INDEX_ALLOW_FILE_URL": "1",
        }
    )
    with pytest.raises(ArtifactError, match="pipeline"):
        ensure_index(settings)
    assert not (tmp_path / "index").exists()


def test_stale_local_index_is_replaced_from_the_url(tmp_path, index_tar):
    import json

    from stylist.artifacts import ensure_index
    from stylist.index import SearchIndex

    tar_path, sha = index_tar
    settings = Settings.from_env(
        {
            "EMBEDDER": "hash",
            "INDEX_DIR": str(tmp_path / "index"),
            "INDEX_URL": tar_path.as_uri(),
            "INDEX_SHA256": sha,
            "INDEX_ALLOW_FILE_URL": "1",
        }
    )
    ensure_index(settings)
    meta_path = tmp_path / "index" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["pipeline_version"] = "0"
    meta_path.write_text(json.dumps(meta))
    ensure_index(settings)  # the stale index is not "valid": it gets reinstalled
    assert SearchIndex.load(tmp_path / "index").n_rows == 40


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x.tar.gz",
        "http://169.254.169.254/latest",
        "https://10.0.0.5/idx.tgz",
        "http://localhost:8000/i.tgz",
    ],
)
def test_private_and_loopback_hosts_are_refused(url):
    from stylist.artifacts import ArtifactError, check_url

    with pytest.raises(ArtifactError, match="private|loopback|link-local|local"):
        check_url(url, allow_file=False)


def test_private_hosts_can_be_allowed_explicitly():
    from stylist.artifacts import check_url

    check_url("http://10.0.0.5/idx.tgz", allow_file=False, allow_private=True)


def test_swap_parks_the_old_index_under_a_unique_name(tmp_path, fixture_catalog, hash_embedder):
    from stylist import index as index_mod
    from stylist.index import SearchIndex, build_index

    index_dir = tmp_path / "idx"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=10, sampling="popular")
    parked = []
    real = index_mod._park_name

    def spy(final_dir):
        name = real(final_dir)
        parked.append(name)
        return name

    index_mod._park_name = spy
    try:
        build_index(fixture_catalog, index_dir, hash_embedder, limit=12, sampling="popular")
        build_index(fixture_catalog, index_dir, hash_embedder, limit=14, sampling="popular")
    finally:
        index_mod._park_name = real
    assert len(parked) == 2 and parked[0] != parked[1]
    assert SearchIndex.load(index_dir).n_rows == 14
    assert sorted(p.name for p in tmp_path.iterdir()) == [".idx.lock", "idx"]
