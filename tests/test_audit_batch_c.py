"""Regression tests for the audit round on artifact fetching and the HTTP surface."""

import io
import tarfile
import threading
import time

import pytest
from fastapi.testclient import TestClient

from stylist.api import create_app
from stylist.artifacts import ArtifactError, check_url, ensure_index, safe_extract
from stylist.config import ConfigError, Settings
from stylist.service import RecommendationService

# ----------------------------------------------------------------------------- artifacts


def _settings(tmp_path, **env):
    return Settings.from_env({"EMBEDDER": "hash", "INDEX_DIR": str(tmp_path / "index"), **env})


@pytest.mark.parametrize(
    "url", ["ftp://host/index.tar.gz", "gopher://x", "data:,abc", "index.tar.gz"]
)
def test_only_http_and_https_urls_are_accepted(url):
    with pytest.raises(ArtifactError, match="scheme"):
        check_url(url, allow_file=False)


def test_file_urls_need_the_explicit_opt_in(tmp_path, index_tar):
    tar_path, sha = index_tar
    with pytest.raises(ArtifactError, match="INDEX_ALLOW_FILE_URL"):
        ensure_index(_settings(tmp_path, INDEX_URL=tar_path.as_uri(), INDEX_SHA256=sha))
    ensure_index(
        _settings(tmp_path, INDEX_URL=tar_path.as_uri(), INDEX_SHA256=sha, INDEX_ALLOW_FILE_URL="1")
    )
    assert (tmp_path / "index" / "meta.json").exists()


def test_redirects_are_checked_with_the_same_rules():
    from stylist.artifacts import _RedirectGuard

    guard = _RedirectGuard(allow_file=False)
    assert guard.check("https://cdn.example/index.tar.gz") is None
    with pytest.raises(ArtifactError, match="scheme"):
        guard.check("file:///etc/passwd")


def _tar(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data, mode in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_archive_with_too_many_members_is_rejected(tmp_path):
    tar_path = tmp_path / "many.tar.gz"
    tar_path.write_bytes(_tar([(f"index/f{i}.npy", b"x", 0o644) for i in range(2001)]))
    with pytest.raises(ArtifactError, match="members"):
        safe_extract(tar_path, tmp_path / "out", max_bytes=10_000_000, max_members=2000)


def test_archive_member_with_unexpected_suffix_is_rejected(tmp_path):
    tar_path = tmp_path / "odd.tar.gz"
    tar_path.write_bytes(_tar([("index/run.sh", b"#!/bin/sh\n", 0o755)]))
    with pytest.raises(ArtifactError, match="file type"):
        safe_extract(tar_path, tmp_path / "out", max_bytes=10_000)


def test_archive_member_nested_too_deep_is_rejected(tmp_path):
    tar_path = tmp_path / "deep.tar.gz"
    tar_path.write_bytes(_tar([("a/b/c/d/e/f/g/h/i/meta.json", b"{}", 0o644)]))
    with pytest.raises(ArtifactError, match="deep"):
        safe_extract(tar_path, tmp_path / "out", max_bytes=10_000)


def test_duplicate_archive_members_are_rejected(tmp_path):
    tar_path = tmp_path / "dup.tar.gz"
    tar_path.write_bytes(
        _tar([("index/meta.json", b"{}", 0o644), ("index/meta.json", b"{}", 0o644)])
    )
    with pytest.raises(ArtifactError, match="twice"):
        safe_extract(tar_path, tmp_path / "out", max_bytes=10_000)


def test_extracted_files_get_fixed_permissions_not_the_archives(tmp_path):
    tar_path = tmp_path / "mode.tar.gz"
    tar_path.write_bytes(_tar([("index/meta.json", b"{}", 0o777)]))
    safe_extract(tar_path, tmp_path / "out", max_bytes=10_000)
    mode = (tmp_path / "out" / "index" / "meta.json").stat().st_mode & 0o777
    assert mode == 0o644


def test_lock_file_survives_the_install(tmp_path, index_tar):
    tar_path, sha = index_tar
    ensure_index(
        _settings(tmp_path, INDEX_URL=tar_path.as_uri(), INDEX_SHA256=sha, INDEX_ALLOW_FILE_URL="1")
    )
    assert (tmp_path / ".index.lock").exists()  # never unlinked: unlinking races a second worker


def test_cli_download_enforces_a_size_cap_and_cleans_up(tmp_path, monkeypatch):
    from stylist import cli

    class FakeResp:
        headers = {"Content-Length": "50"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return (
                b"x" * 50
                if not getattr(self, "done", False) and not setattr(self, "done", True)
                else b""
            )

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda url, timeout=60: FakeResp())
    out = tmp_path / "raw.jsonl.gz"
    with pytest.raises(cli.DownloadError, match="bytes"):
        cli._download("https://example/x", out, max_bytes=10)
    assert not out.exists() and not out.with_suffix(out.suffix + ".part").exists()


# ----------------------------------------------------------------------------- api


def _app(fixture_index, hash_embedder, **env):
    settings = Settings.from_env({"EMBEDDER": "hash", **env})
    svc = RecommendationService(fixture_index, hash_embedder, settings, llm=None)
    return create_app(settings, service=svc)


def test_cors_is_off_unless_origins_are_configured(fixture_index, hash_embedder):
    headers = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    with TestClient(_app(fixture_index, hash_embedder)) as c:
        r = c.options("/recommend", headers=headers)
        assert "access-control-allow-origin" not in r.headers
    with TestClient(
        _app(fixture_index, hash_embedder, CORS_ALLOW_ORIGINS="https://app.example")
    ) as c:
        r = c.options("/recommend", headers={**headers, "Origin": "https://app.example"})
        assert r.headers.get("access-control-allow-origin") == "https://app.example"
        r = c.options("/recommend", headers=headers)
        assert "access-control-allow-origin" not in r.headers


def test_rate_limit_returns_429_with_retry_after(fixture_index, hash_embedder):
    with TestClient(_app(fixture_index, hash_embedder, RATE_LIMIT_PER_MINUTE="3")) as c:
        codes = [
            c.post("/recommend", json={"query": "boots", "k": 1}).status_code for _ in range(4)
        ]
        assert codes[:3] == [200, 200, 200] and codes[3] == 429
        r = c.post("/recommend", json={"query": "boots", "k": 1})
        assert r.json()["error"]["code"] == "rate_limited" and r.headers.get("retry-after")
        assert c.get("/health").status_code == 200  # ops endpoints are not rate limited


def test_rate_limit_zero_disables_it(fixture_index, hash_embedder):
    with TestClient(_app(fixture_index, hash_embedder, RATE_LIMIT_PER_MINUTE="0")) as c:
        assert all(
            c.post("/recommend", json={"query": "boots"}).status_code == 200 for _ in range(5)
        )


def test_inflight_cap_returns_503_busy(fixture_index, hash_embedder):
    app = _app(fixture_index, hash_embedder, MAX_INFLIGHT_REQUESTS="1", RATE_LIMIT_PER_MINUTE="0")
    svc = app.state.service if hasattr(app.state, "service") else None
    with TestClient(app) as c:
        svc = c.app.state.service
        real = svc.recommend
        gate = threading.Event()

        async def slow(req):
            import asyncio

            await asyncio.get_running_loop().run_in_executor(None, gate.wait, 5)
            return await real(req)

        svc.recommend = slow
        results = {}

        def first():
            results["first"] = c.post("/recommend", json={"query": "boots"}).status_code

        t = threading.Thread(target=first)
        t.start()
        time.sleep(0.3)
        second = c.post("/recommend", json={"query": "boots"})
        gate.set()
        t.join()
        assert second.status_code == 503 and second.json()["error"]["code"] == "busy"
        assert results["first"] == 200


def test_oversized_body_is_a_413(fixture_index, hash_embedder):
    with TestClient(_app(fixture_index, hash_embedder, MAX_BODY_BYTES="200")) as c:
        r = c.post("/recommend", json={"query": "x" * 300})
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
        r = c.post(
            "/recommend", content=b"{" + b" " * 300, headers={"content-type": "application/json"}
        )
        assert r.status_code == 413


def test_startup_error_is_curated_and_redacts_secrets(tmp_path):
    settings = Settings.from_env(
        {"EMBEDDER": "hash", "INDEX_DIR": str(tmp_path / "nope"), "LLM_PROVIDER": "none"}
    )
    with TestClient(create_app(settings)) as c:
        msg = c.get("/health").json()["load_error"]
        assert "nope" not in msg and str(tmp_path) not in msg
        assert msg.startswith("index")  # a curated sentence, not a stack trace

    from stylist.api import _public_error

    assert "sk-ant-" not in _public_error(RuntimeError("bad key sk-ant-api03-abcdefghijklmnop"))
    assert "api_key=" not in _public_error(RuntimeError("api_key=abcdefghijklmnop1234"))


def test_startup_fail_fast_raises_instead_of_serving_503(tmp_path):
    settings = Settings.from_env(
        {"EMBEDDER": "hash", "INDEX_DIR": str(tmp_path / "nope"), "STARTUP_FAIL_FAST": "1"}
    )
    with pytest.raises(Exception, match="index"):
        with TestClient(create_app(settings)):
            pass


def test_log_level_is_validated():
    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        Settings.from_env({"EMBEDDER": "hash", "LOG_LEVEL": "LOUD"}).validate()


def test_serve_binds_loopback_by_default():
    from stylist.cli import build_parser

    args = build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"


# ----------------------------------------------------------------------------- usage accounting


async def test_llm_usage_is_counted_per_request_and_reported(fixture_index, hash_embedder):
    from stylist.llm import FakeLLM, usage_scope
    from stylist.planner import PlannerOutput
    from stylist.reranker import SlotRerankOutput
    from stylist.schemas import RecommendRequest

    def handler(system, user, schema):
        if schema is PlannerOutput:
            return PlannerOutput(
                intent="boots",
                slots=[{"name": "boots", "search_query": "snow boots", "keywords": ["boot"]}],
            )
        return SlotRerankOutput(picks=[], no_good_match=False, note="")

    llm = FakeLLM(handler=handler, usage=(120, 30))  # every call reports 120 in / 30 out
    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env({"EMBEDDER": "hash", "PLAN_CACHE_SIZE": "0"}),
        llm=llm,
    )
    resp = await svc.recommend(RecommendRequest(query="snow boots", k=2))
    assert resp.llm_info.calls == 2  # one plan + one rerank
    assert resp.llm_info.input_tokens == 240 and resp.llm_info.output_tokens == 60

    with usage_scope() as usage:  # scopes nest per task: a second request starts at zero
        await llm.complete_json(system="s", user="u", schema=PlannerOutput)
    assert usage.calls == 1 and usage.input_tokens == 120


def test_prices_are_rounded_to_cents(fixture_index, hash_embedder):
    from stylist.retrieval import Candidate, Retriever

    r = Retriever(fixture_index, hash_embedder, Settings.from_env({"EMBEDDER": "hash"}))
    old = fixture_index.catalog.loc[0, "price"]
    fixture_index.catalog.loc[0, "price"] = 8.99
    try:
        assert (
            r._hydrate(Candidate(0, 0, 0.0)).price == 8.99
        )  # float32 storage would give 8.98999977
    finally:
        fixture_index.catalog.loc[0, "price"] = old
