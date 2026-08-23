"""Offline evaluation of the recommendation pipeline.

Runs the 20 queries in scripts/eval_queries.json through several configurations and
reports:

  keyword_match@k   share of returned items whose title passes the hand written type
                    rules for their slot (a regression diagnostic for product type, it
                    says nothing about style or taste); whole-word matches only
  macro_match       the same, averaged per query, with a bootstrap 95% interval
  slot_recall       share of the expected slots (rule names) that a returned slot mapped to
  unmapped_slots    returned slots that matched no rule name (scored against the union)
  query_success     queries where every returned slot has a matching item and none is empty
  price_violations  items with a known price outside an explicit min/max price
  inferred_over     items priced above a budget that was only in the sentence (a warning,
                    not a violation: the planner may read the sentence differently)
  p50 / p95 ms      wall clock per request

Configs compare retrieval channels (bm25 / dense / hybrid), the keyword boost and the
rating prior separately, and the LLM planner + reranker. All llm configs share one plan
cache, so their comparison is paired: same plans, different retrieval or reranking.
Needs a built index (INDEX_DIR) and, for the llm configs, an LLM key in the environment.

usage: uv run python scripts/evaluate.py [--index-dir data/index] [--configs hybrid,llm]
           [--out docs/eval_results.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
import time
from collections import OrderedDict
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
    "hybrid_nokw": ({"keyword_boost": 0.0}, {"use_llm": False}),
    "hybrid_noquality": ({"quality_weight": 0.0}, {"use_llm": False}),
    "hybrid_noboost": ({"keyword_boost": 0.0, "quality_weight": 0.0}, {"use_llm": False}),
    "llm_plan": ({}, {"use_llm": True, "rerank": False}),
    "llm_plan_dense": ({"channels": ("dense",)}, {"use_llm": True, "rerank": False}),
    "llm_plan_bm25": ({"channels": ("bm25",)}, {"use_llm": True, "rerank": False}),
    "llm_plan_rerank": ({}, {"use_llm": True, "rerank": True}),
}


def _term_rx(term: str) -> re.Pattern:
    """Whole-word match, plural tolerant; multi-word terms match with any separator."""
    words = [re.escape(w) for w in term.lower().split()]
    body = r"[\s\-]+".join(words)
    return re.compile(rf"\b{body}(?:s|es)?\b")


def _passes(title: str, rule: dict) -> bool:
    """`none`: no listed term may appear. `all`: a required part (a brand, spelled any of
    the listed ways): one of them must appear. `any`: the product type: one must appear."""
    low = title.lower()
    if any(_term_rx(n).search(low) for n in rule.get("none", [])):
        return False
    if rule.get("all") and not any(_term_rx(a).search(low) for a in rule["all"]):
        return False
    return any(_term_rx(a).search(low) for a in rule.get("any", []))


def score_query(rules: dict, returned: list[tuple[str, list[str]]]) -> dict:
    """Score one query: `rules` maps expected slot name -> rule, `returned` is a list of
    (slot name, item titles). Returns counts for every metric in the module docstring."""
    union = {"any": sorted({a for r in rules.values() for a in r.get("any", [])})}
    found: set[int] = set()
    items = match = mapped_items = mapped_match = unmapped = empty = 0
    success = True
    for name, titles in returned:
        rule = _match_slot_rule(name, rules)
        if rule is None:
            unmapped += 1
            rule = union
        else:
            found.add(id(rule))
        if not titles:
            empty += 1
            success = False
            continue
        hits = sum(_passes(t, rule) for t in titles)
        items += len(titles)
        match += hits
        if rule is not union:
            mapped_items += len(titles)
            mapped_match += hits
        if hits == 0:
            success = False
    return {
        "expected_slots": len(rules),
        "slots_found": len(found),
        "unmapped_slots": unmapped,
        "items": items,
        "match": match,
        "mapped_items": mapped_items,
        "mapped_match": mapped_match,
        "empty": empty,
        "success": success,
    }


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% interval of the mean over queries."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return (round(means[int(0.025 * n)], 3), round(means[int(0.975 * n) - 1], 3))


def _match_slot_rule(slot_name: str, rules: dict) -> dict | None:
    """Find the rule for a returned slot by name overlap; falls back to the union of all."""
    name_words = set(re.findall(r"[a-z]+", slot_name.lower()))
    for rule_name, rule in rules.items():
        if set(re.findall(r"[a-z]+", rule_name.lower())) & name_words:
            return rule
    return None


async def run_config(
    name: str,
    queries: list[dict],
    index,
    embedder,
    base: Settings,
    llm,
    plan_cache: OrderedDict | None = None,
):
    s_over, r_over = CONFIGS[name]
    settings = replace(base, **s_over)
    settings.validate()
    svc = RecommendationService(
        index,
        embedder,
        settings,
        llm if r_over.get("use_llm") else None,
        plan_cache=plan_cache if r_over.get("use_llm") else None,
    )

    rows = []
    failures = []
    for q in queries:
        req = RecommendRequest(
            query=q["query"],
            k=4,
            max_price=q.get("max_price"),
            min_price=q.get("min_price"),
            **r_over,
        )
        t = time.perf_counter()
        try:
            res = await svc.recommend(req)
        except Exception as exc:  # noqa: BLE001 - record and keep going
            failures.append({"id": q["id"], "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        ms = (time.perf_counter() - t) * 1000
        scored = score_query(
            q["slots"], [(s.name, [it.title for it in s.items]) for s in res.slots]
        )
        price_bad = inferred_over = 0
        for slot in res.slots:
            for it in slot.items:
                if it.price is None:
                    continue
                if q.get("max_price") is not None and it.price > q["max_price"]:
                    price_bad += 1
                if q.get("min_price") is not None and it.price < q["min_price"]:
                    price_bad += 1
                if q.get("budget_in_text") is not None and it.price > q["budget_in_text"]:
                    inferred_over += 1
        rows.append(
            {
                "id": q["id"],
                "slots": len(res.slots),
                **scored,
                "price_bad": price_bad,
                "inferred_over": inferred_over,
                "ms": round(ms, 1),
                "planner": res.llm_info.planner_used,
                "rerank": res.llm_info.rerank_used,
                "llm_calls": res.llm_info.calls,
                "tokens": [res.llm_info.input_tokens, res.llm_info.output_tokens],
                "warnings": len(res.warnings),
                "titles": [[it.title[:70] for it in slot.items] for slot in res.slots],
                "slot_names": [slot.name for slot in res.slots],
            }
        )
    items = sum(r["items"] for r in rows) or 1
    mapped = sum(r["mapped_items"] for r in rows) or 1
    per_query = [r["match"] / r["items"] for r in rows if r["items"]]
    lat = [r["ms"] for r in rows]
    lo, hi = bootstrap_ci(per_query)
    summary = {
        "config": name,
        "queries": len(rows),
        "failures": failures,
        "keyword_match_at_k": round(sum(r["match"] for r in rows) / items, 3),
        "macro_match": round(statistics.fmean(per_query), 3) if per_query else 0.0,
        "macro_match_ci95": [lo, hi],
        "mapped_precision": round(sum(r["mapped_match"] for r in rows) / mapped, 3),
        "slot_recall": round(
            sum(r["slots_found"] for r in rows) / max(1, sum(r["expected_slots"] for r in rows)),
            3,
        ),
        "unmapped_slots": sum(r["unmapped_slots"] for r in rows),
        "query_success": round(sum(1 for r in rows if r["success"]) / max(1, len(rows)), 3),
        "price_violations": sum(r["price_bad"] for r in rows),
        "inferred_over_budget": sum(r["inferred_over"] for r in rows),
        "empty_slots": sum(r["empty"] for r in rows),
        "total_slots": sum(r["slots"] for r in rows),
        "llm_calls": sum(r["llm_calls"] for r in rows),
        "tokens": [sum(r["tokens"][0] for r in rows), sum(r["tokens"][1] for r in rows)],
        "p50_ms": round(statistics.median(lat), 1) if lat else 0.0,
        "p95_ms": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 1) if lat else 0.0,
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
    from stylist.llm.prompts import PROMPT_VERSION

    out = {
        "index": {
            "dir": Path(base.index_dir).name,
            "rows": index.n_rows,
            "sampling": index.meta.sampling,
            "built_at": index.meta.built_at,
            "pipeline_version": index.meta.pipeline_version,
        },
        "llm": {
            "provider": llm.provider if llm else None,
            "model": llm.model if llm else None,
            "prompt_version": PROMPT_VERSION,
        },
        "queries": len(queries),
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [],
    }
    plan_cache: OrderedDict = OrderedDict()  # shared by every llm config: paired plans
    cache_path = Path(args.plan_cache) if args.plan_cache else None
    if cache_path and cache_path.exists():
        from stylist.planner import QueryPlan

        for key, plan in json.loads(cache_path.read_text()):
            plan_cache[tuple(key)] = QueryPlan.model_validate(plan)
        print(f"loaded {len(plan_cache)} cached plans from {cache_path}", file=sys.stderr)
    for name in names:
        if name.startswith("llm") and llm is None:
            print(f"skip {name}: no llm configured", file=sys.stderr)
            continue
        print(f"running {name} ...", file=sys.stderr)
        summary = await run_config(name, queries, index, embedder, base, llm, plan_cache)
        out["results"].append(summary)
        print(
            f"{name:16s} match@k={summary['keyword_match_at_k']:.3f} "
            f"macro={summary['macro_match']:.3f} {summary['macro_match_ci95']} "
            f"recall={summary['slot_recall']:.2f} success={summary['query_success']:.2f} "
            f"empty={summary['empty_slots']}/{summary['total_slots']} "
            f"price_bad={summary['price_violations']} fail={len(summary['failures'])} "
            f"p50={summary['p50_ms']}ms p95={summary['p95_ms']}ms",
            file=sys.stderr,
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)
    if cache_path:
        cache_path.write_text(
            json.dumps([[list(k), p.model_dump(mode="json")] for k, p in plan_cache.items()])
        )
        print(f"saved {len(plan_cache)} plans to {cache_path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir")
    p.add_argument(
        "--configs",
        default="bm25,dense,hybrid,hybrid_nokw,hybrid_noquality,llm_plan,llm_plan_rerank",
    )
    p.add_argument("--out", default="docs/eval_results.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--plan-cache",
        default=None,
        help="json file to load/save LLM plans, so runs on different indexes use the same plans",
    )
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
