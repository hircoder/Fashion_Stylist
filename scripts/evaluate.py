"""Offline evaluation of the recommendation pipeline.

Runs the 20 queries in scripts/eval_queries.json through several configurations and
reports:

  keyword_match@k   share of returned items whose title passes the hand written type
                    rules for their slot (a regression diagnostic for product type, it
                    says nothing about style or taste)
  price_violations  items with a known price outside an explicit max_price
  empty_slots       slots that came back with no item
  p50 / p95 ms      wall clock per request

Configs compare retrieval channels (bm25 / dense / hybrid), the keyword boost, and the
LLM planner + reranker. Needs a built index (INDEX_DIR) and, for the llm configs, an
LLM key in the environment.

usage: uv run python scripts/evaluate.py [--index-dir data/index] [--configs hybrid,llm]
           [--out docs/eval_results.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
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
from stylist.llm import make_llm_client  # noqa: E402
from stylist.schemas import RecommendRequest  # noqa: E402
from stylist.service import RecommendationService  # noqa: E402

CONFIGS = {
    # name: (settings overrides, request overrides)
    "bm25": ({"channels": ("bm25",)}, {"use_llm": False}),
    "dense": ({"channels": ("dense",)}, {"use_llm": False}),
    "hybrid": ({}, {"use_llm": False}),
    "hybrid_noboost": ({"keyword_boost": 0.0, "quality_weight": 0.0}, {"use_llm": False}),
    "llm_plan": ({}, {"use_llm": True, "rerank": False}),
    "llm_plan_dense": ({"channels": ("dense",)}, {"use_llm": True, "rerank": False}),
    "llm_plan_bm25": ({"channels": ("bm25",)}, {"use_llm": True, "rerank": False}),
    "llm_plan_rerank": ({}, {"use_llm": True, "rerank": True}),
}


def _passes(title: str, rule: dict) -> bool:
    low = title.lower()
    if rule.get("none") and any(n in low for n in rule["none"]):
        return False
    return any(a in low for a in rule.get("any", []))


def _match_slot_rule(slot_name: str, rules: dict) -> dict | None:
    """Find the rule for a returned slot by name overlap; falls back to the union of all."""
    name = slot_name.lower()
    for rule_name, rule in rules.items():
        words = re.findall(r"[a-z]+", rule_name.lower())
        if any(w in name for w in words) or any(w in rule_name.lower() for w in name.split()):
            return rule
    return None


async def run_config(name: str, queries: list[dict], index, embedder, base: Settings, llm):
    s_over, r_over = CONFIGS[name]
    settings = replace(base, **s_over)
    svc = RecommendationService(index, embedder, settings, llm if r_over.get("use_llm") else None)

    rows = []
    for q in queries:
        req = RecommendRequest(query=q["query"], k=4, max_price=q.get("max_price"), **r_over)
        t = time.perf_counter()
        res = await svc.recommend(req)
        ms = (time.perf_counter() - t) * 1000
        all_rules = q["slots"]
        union = {"any": sorted({a for r in all_rules.values() for a in r.get("any", [])})}
        n_items = n_match = n_price_bad = empty = 0
        for slot in res.slots:
            rule = _match_slot_rule(slot.name, all_rules) or union
            if not slot.items:
                empty += 1
            for it in slot.items:
                n_items += 1
                n_match += _passes(it.title, rule)
                if q.get("max_price") and it.price is not None and it.price > q["max_price"]:
                    n_price_bad += 1
        rows.append(
            {
                "id": q["id"],
                "slots": len(res.slots),
                "items": n_items,
                "match": n_match,
                "price_bad": n_price_bad,
                "empty": empty,
                "ms": round(ms, 1),
                "planner": res.llm_info.planner_used,
                "rerank": res.llm_info.rerank_used,
                "titles": [[it.title[:70] for it in slot.items] for slot in res.slots],
                "slot_names": [slot.name for slot in res.slots],
            }
        )
    items = sum(r["items"] for r in rows) or 1
    lat = [r["ms"] for r in rows]
    summary = {
        "config": name,
        "queries": len(rows),
        "keyword_match_at_k": round(sum(r["match"] for r in rows) / items, 3),
        "price_violations": sum(r["price_bad"] for r in rows),
        "empty_slots": sum(r["empty"] for r in rows),
        "total_slots": sum(r["slots"] for r in rows),
        "p50_ms": round(statistics.median(lat), 1),
        "p95_ms": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 1),
        "rows": rows,
    }
    return summary


async def main_async(args):
    base = Settings.from_env()
    if args.index_dir:
        base = replace(base, index_dir=Path(args.index_dir))
    index = SearchIndex.load(base.index_dir, expected_model=base.embedding_name)
    embedder = make_embedder(base)
    llm = make_llm_client(base)
    queries = json.loads((ROOT / "scripts" / "eval_queries.json").read_text())["queries"]
    if args.limit:
        queries = queries[: args.limit]
    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    out = {
        "index": {
            "dir": str(base.index_dir),
            "rows": index.n_rows,
            "sampling": index.meta.sampling,
        },
        "llm": {"provider": llm.provider if llm else None, "model": llm.model if llm else None},
        "results": [],
    }
    for name in names:
        if name.startswith("llm") and llm is None:
            print(f"skip {name}: no llm configured", file=sys.stderr)
            continue
        print(f"running {name} ...", file=sys.stderr)
        summary = await run_config(name, queries, index, embedder, base, llm)
        out["results"].append(summary)
        print(
            f"{name:16s} match@k={summary['keyword_match_at_k']:.3f} "
            f"empty={summary['empty_slots']}/{summary['total_slots']} "
            f"price_bad={summary['price_violations']} "
            f"p50={summary['p50_ms']}ms p95={summary['p95_ms']}ms",
            file=sys.stderr,
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir")
    p.add_argument("--configs", default="bm25,dense,hybrid,hybrid_noboost,llm_plan,llm_plan_rerank")
    p.add_argument("--out", default="docs/eval_results.json")
    p.add_argument("--limit", type=int, default=None)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
