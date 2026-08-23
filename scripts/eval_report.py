"""Turn eval json files (from scripts/evaluate.py) into markdown tables.

usage: uv run python scripts/eval_report.py docs/eval_popular100k.json docs/eval_full.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def table(path: Path) -> str:
    data = json.loads(path.read_text())
    idx = data["index"]
    lines = [
        f"index: `{idx['dir']}` ({idx['rows']:,} rows, sampling={idx['sampling']}), "
        f"llm: {data['llm']['model'] or 'none'}, prompt v{data['llm'].get('prompt_version', '?')}, "
        f"{data.get('queries', '?')} queries, code {data.get('git_sha') or '?'}",
        "",
        "| config | match@k | macro (95% CI) | mapped precision | slot recall | query success "
        "| empty slots | price viol. | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in data["results"]:
        lo, hi = r.get("macro_match_ci95", [0, 0])
        lines.append(
            f"| {r['config']} | {r['keyword_match_at_k']:.3f} "
            f"| {r.get('macro_match', 0):.3f} ({lo:.2f} to {hi:.2f}) "
            f"| {r.get('mapped_precision', 0):.3f} | {r.get('slot_recall', 0):.2f} "
            f"| {r.get('query_success', 0):.2f} "
            f"| {r['empty_slots']}/{r['total_slots']} | {r['price_violations']} "
            f"| {r['p50_ms']:.0f} | {r['p95_ms']:.0f} |"
        )
    deltas = data.get("paired_deltas") or {}
    if deltas:
        lines += [
            "",
            "paired differences on the same queries (mean of per-query match rate, "
            "95% bootstrap interval):",
            "",
        ]
        lines += ["| comparison | mean delta | 95% CI | n |", "|---|---|---|---|"]
        for name, d in deltas.items():
            lines.append(
                f"| {name} | {d['mean']:+.3f} | {d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f} "
                f"| {d['n']} |"
            )
    return "\n".join(lines)


def per_query(path: Path, config: str) -> str:
    data = json.loads(path.read_text())
    res = next(r for r in data["results"] if r["config"] == config)
    lines = ["| query | slots | items | matched | ms |", "|---|---|---|---|---|"]
    for row in res["rows"]:
        lines.append(
            f"| {row['id']} | {', '.join(row['slot_names'])[:60]} | {row['items']} | "
            f"{row['match']} | {row['ms']:.0f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(f"### {Path(arg).stem}\n")
        print(table(Path(arg)))
        print()
