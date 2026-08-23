"""Offline evaluation of the recommendation pipeline.

Runs the 28 queries in scripts/eval_queries.json through several configurations and
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
import math
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
    # a slot the planner invented is scored against the union of the query's rules: every
    # type word counts, every forbidden word still forbids, a required brand stays required
    union: dict = {"any": sorted({a for r in rules.values() for a in r.get("any", [])})}
    nones = sorted({n for r in rules.values() for n in r.get("none", [])})
    alls = sorted({a for r in rules.values() for a in r.get("all", [])})
    if nones:
        union["none"] = nones
    if alls:
        union["all"] = alls
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
    if len(found) < len(rules):
        success = False  # an expected slot never came back
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


def paired_delta(a: dict[str, float], b: dict[str, float], n: int = 1000, seed: int = 0) -> dict:
    """Mean of (a - b) over the queries both have, with a percentile bootstrap 95% interval:
    the honest way to say one configuration beats another on the same queries."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n": 0, "mean": 0.0, "ci95": [0.0, 0.0]}
    diffs = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return {
        "n": len(keys),
        "mean": round(statistics.fmean(diffs), 3),
        "ci95": [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n) - 1], 3)],
    }


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% interval of the mean over queries."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return (round(means[int(0.025 * n)], 3), round(means[int(0.975 * n) - 1], 3))


def _match_slot_rule(slot_name: str, rules: dict) -> dict | None:
    """Find the rule for a returned slot: an exact name first, then the rule whose name
    shares the most words (Jaccard) with it; None when nothing overlaps."""
    name = slot_name.lower().strip()
    if name in rules:
        return rules[name]
    name_words = set(re.findall(r"[a-z]+", name))
    best, best_score = None, 0.0
    for rule_name, rule in rules.items():
        words = set(re.findall(r"[a-z]+", rule_name.lower()))
        inter = len(words & name_words)
        if not inter:
            continue
        score = inter / len(words | name_words)
        if score > best_score:
            best, best_score = rule, score
    return best


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
    try:
        return await _run_config_inner(name, queries, svc, r_over)
    finally:
        svc.close()  # each configuration gets its own thread pool, give it back


async def _run_config_inner(name: str, queries: list[dict], svc, r_over: dict):
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
            # a failed request is a failed query: it stays in every denominator as a zero
            rows.append(
                {
                    "id": q["id"],
                    "slots": 0,
                    "expected_slots": len(q["slots"]),
                    "slots_found": 0,
                    "unmapped_slots": 0,
                    "items": 0,
                    "match": 0,
                    "mapped_items": 0,
                    "mapped_match": 0,
                    "empty": 0,
                    "success": False,
                    "failed": True,
                    "price_bad": 0,
                    "inferred_over": 0,
                    "ms": round((time.perf_counter() - t) * 1000, 1),
                    "planner": None,
                    "rerank": False,
                    "llm_calls": 0,
                    "tokens": [0, 0],
                    "warnings": 0,
                    "titles": [],
                    "slot_names": [],
                }
            )
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
    # per-query rate; a query with no items (failed, or every slot empty) scores zero
    per_query = [(r["match"] / r["items"]) if r["items"] else 0.0 for r in rows]
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
        "p95_ms": round(sorted(lat)[max(0, math.ceil(len(lat) * 0.95) - 1)], 1) if lat else 0.0,
        "per_query_match": {r["id"]: round(v, 4) for r, v in zip(rows, per_query, strict=True)},
        "rows": rows,
    }
    return summary


def _git_sha() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
        )
        return out.stdout.strip() or None
    except OSError:
        return None


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
        "git_sha": _git_sha(),
        "schema_version": 2,
        "results": [],
        "paired_deltas": {},
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
    # paired comparisons on the same queries: every llm config against llm_plan, every
    # retrieval-only config against hybrid
    by_name = {r["config"]: r["per_query_match"] for r in out["results"]}
    for name, per in by_name.items():
        baseline = "llm_plan" if name.startswith("llm") else "hybrid"
        if baseline in by_name and baseline != name:
            out["paired_deltas"][f"{name} - {baseline}"] = paired_delta(per, by_name[baseline])
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
