"""Serving footprint of an index: memory after load, and retrieval latency (no LLM).

usage: uv run python scripts/benchmark.py --index-dir data/index \
           [--concurrency 1,2,4] [--requests 40] [--device cpu]

--device cpu makes the laptop numbers comparable with a CPU-only container.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import resource
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylist.config import Settings  # noqa: E402
from stylist.embeddings import make_embedder  # noqa: E402
from stylist.index import SearchIndex  # noqa: E402
from stylist.schemas import RecommendRequest  # noqa: E402
from stylist.service import RecommendationService  # noqa: E402

QUERIES = [
    "warm waterproof boots for hiking in snow",
    "elegant black dress for a wedding guest",
    "men's slim fit chinos for the office",
    "cozy oversized sweater for fall",
    "running shoes with arch support",
    "something to keep my ears warm in winter",
    "comfortable sandals for walking all day",
    "white sneakers under $40",
]


def rss_mb() -> float:
    """Peak resident set in MB. ru_maxrss is bytes on macOS and kilobytes on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if platform.system() == "Darwin" else raw / 1024


async def run(svc: RecommendationService, concurrency: int, n_requests: int) -> dict:
    reqs = [
        RecommendRequest(query=QUERIES[i % len(QUERIES)], use_llm=False) for i in range(n_requests)
    ]
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []

    async def one(r):
        async with sem:
            t = time.perf_counter()
            await svc.recommend(r)
            lat.append((time.perf_counter() - t) * 1000)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(r) for r in reqs))
    wall = time.perf_counter() - t0
    lat.sort()
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "p50_ms": round(statistics.median(lat), 1),
        "p95_ms": round(lat[max(0, int(len(lat) * 0.95) - 1)], 1),
        "throughput_rps": round(n_requests / wall, 1),
    }


async def main_async(args) -> dict:
    base = replace(
        Settings.from_env(),
        index_dir=Path(args.index_dir),
        embed_device=args.device or Settings.from_env().embed_device,
    )
    before = rss_mb()
    t = time.perf_counter()
    index = SearchIndex.load(base.index_dir, expected_model=base.embedding_name)
    load_s = time.perf_counter() - t
    embedder = make_embedder(base)
    svc = RecommendationService(index, embedder, base, llm=None)  # one service, one loop
    await svc.recommend(RecommendRequest(query="warm up", use_llm=False))
    after = rss_mb()
    out = {
        "index_dir": str(base.index_dir),
        "rows": index.n_rows,
        "sampling": index.meta.sampling,
        "device": args.device or "default",
        "load_seconds": round(load_s, 1),
        "rss_mb_after_load": round(after, 0),
        "rss_mb_delta": round(after - before, 0),
        "runs": [],
    }
    for c in [int(x) for x in args.concurrency.split(",")]:
        out["runs"].append(await run(svc, c, args.requests))
        print(out["runs"][-1], file=sys.stderr)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", default="data/index")
    p.add_argument("--concurrency", default="1,2,4")
    p.add_argument("--requests", type=int, default=40)
    p.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    args = p.parse_args()
    print(json.dumps(asyncio.run(main_async(args)), indent=2))


if __name__ == "__main__":
    main()
