"""The 28 evaluation queries against the LIVE endpoint, scored with the same rules
as the local harness. Two passes: the first one triggers background plans (and is
reported as the cold column), the second one shows steady-state quality.

  python3 deploy/aws/quality_probe.py https://HOST
"""

import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from evaluate import score_query  # noqa: E402


def call(base: str, query: str, k: int = 4) -> dict:
    body = json.dumps({"query": query, "k": k}).encode()
    req = urllib.request.Request(
        base + "/recommend", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def run_pass(base: str, queries: list[dict], label: str) -> dict:
    per_query = []
    for q in queries:
        r = call(base, q["query"])
        returned = [(s["name"], [i["title"] for i in s["items"]]) for s in r["slots"]]
        s = score_query(q["slots"], returned)
        s["planner"] = r["llm_info"]["planner_used"]
        s["match_at_k"] = s["match"] / s["items"] if s["items"] else 0.0
        per_query.append(s)
    macro = statistics.mean(x["match_at_k"] for x in per_query)
    micro = sum(x["match"] for x in per_query) / max(1, sum(x["items"] for x in per_query))
    out = {
        "label": label,
        "n": len(per_query),
        "match_at_k_micro": round(micro, 3),
        "match_at_k_macro": round(macro, 3),
        "query_success": round(sum(x["success"] for x in per_query) / len(per_query), 3),
        "empty_slots": sum(x["empty"] for x in per_query),
        "planner_llm_rate": round(
            sum(x["planner"] == "llm" for x in per_query) / len(per_query), 3
        ),
    }
    print(json.dumps(out))
    return out


def main() -> None:
    base = sys.argv[1].rstrip("/")
    # optional second argument: the experiment file to write (a rerun must never
    # overwrite the record of an earlier one)
    out = sys.argv[2] if len(sys.argv) > 2 else "exp18_tokyo_quality.json"
    queries = json.load(open(Path(__file__).resolve().parents[2] / "scripts/eval_queries.json"))[
        "queries"
    ]
    cold = run_pass(base, queries, "pass1_cold")
    time.sleep(10)  # background plans land in the exact and semantic caches
    warm = run_pass(base, queries, "pass2_planned")
    json.dump(
        {"cold": cold, "warm": warm},
        open(Path(__file__).with_name("experiments") / out, "w"),
        indent=2,
    )


main()
