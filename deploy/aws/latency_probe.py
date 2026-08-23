"""Latency probe: repeated POST /recommend, percentiles per configuration.

The probe measures wall time from this machine, so the network to the edge is part of
the number; run it twice (against CloudFront and against the origin) to split the two.
"""

import argparse
import json
import statistics
import time
import urllib.request

QUERIES = [
    "I need an outfit to go to the beach this summer",
    "warm waterproof boots for hiking in the snow",
    "smart casual outfit for a job interview at a startup",
    "white sneakers under $40",
    "a gift for my 6 year old daughter who loves unicorns",
    "cozy oversized sweater for fall",
]


def call(base, body, timeout=65):
    req = urllib.request.Request(
        base.rstrip("/") + "/recommend",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    ms = (time.perf_counter() - t) * 1000
    return ms, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--use-llm", default="true")
    ap.add_argument("--rerank", default=None, help="unset = server default")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lat, rows = [], []
    for i in range(args.n):
        body = {"query": QUERIES[i % len(QUERIES)], "k": 4, "use_llm": args.use_llm == "true"}
        if args.rerank is not None:
            body["rerank"] = args.rerank == "true"
        try:
            ms, data = call(args.base, body)
        except Exception as exc:
            rows.append({"i": i, "error": str(exc)[:200]})
            continue
        lat.append(ms)
        rows.append(
            {
                "i": i,
                "ms": round(ms, 1),
                "planner": data["llm_info"]["planner_used"],
                "cache_hit": data["llm_info"]["plan_cache_hit"],
                "rerank": data["llm_info"]["rerank_used"],
                "calls": data["llm_info"]["calls"],
                "server_total_ms": data["timings"]["total_ms"],
                "slots": len(data["slots"]),
                "items": sum(len(s["items"]) for s in data["slots"]),
            }
        )
        print(rows[-1])
    lat.sort()
    out = {
        "label": args.label,
        "base": args.base,
        "n": args.n,
        "errors": sum(1 for r in rows if "error" in r),
        "p50_ms": round(statistics.median(lat), 1) if lat else None,
        "p95_ms": round(lat[max(0, -(-len(lat) * 95 // 100) - 1)], 1) if lat else None,
        "min_ms": round(lat[0], 1) if lat else None,
        "max_ms": round(lat[-1], 1) if lat else None,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("label", "p50_ms", "p95_ms", "errors")}))


if __name__ == "__main__":
    main()
