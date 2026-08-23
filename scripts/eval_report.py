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
        f"index: `{Path(idx['dir']).name}` ({idx['rows']:,} rows, sampling={idx['sampling']}), "
        f"llm: {data['llm']['model'] or 'none'}",
        "",
        "| config | keyword_match@k | empty slots | price violations | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for r in data["results"]:
        lines.append(
            f"| {r['config']} | {r['keyword_match_at_k']:.3f} "
            f"| {r['empty_slots']}/{r['total_slots']} | {r['price_violations']} "
            f"| {r['p50_ms']:.0f} | {r['p95_ms']:.0f} |"
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
