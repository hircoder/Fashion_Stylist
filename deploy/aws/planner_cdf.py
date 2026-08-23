"""Times real planner calls from the box itself. The point: how often would a
0.35 s bounded wait actually receive the plan, and what would 0.25 s change?
Run on the instance: /opt/stylist/.venv/bin/python planner_cdf.py
"""

import asyncio
import json
import statistics
import sys
import time

sys.path.insert(0, "/opt/stylist/src")

from stylist.config import Settings  # noqa: E402
from stylist.llm import make_llm_client  # noqa: E402
from stylist.planner import LLMPlanner  # noqa: E402

QUERIES = [
    "I need an outfit to go to the beach this summer",
    "warm waterproof boots for hiking in the snow, under $80",
    "what should my husband wear to an outdoor wedding in june",
    "something cozy for working from home in winter",
    "a gift for my 6 year old daughter who loves unicorns",
    "smart casual outfit for a job interview at a startup",
    "rainy day commute clothes that still look office ready",
    "date night look for a jazz bar, black preferred",
    "lightweight running gear for humid august mornings",
    "capsule wardrobe basics for a first business trip",
    "festival outfit that survives mud and sun both",
    "warm layers for watching soccer outside in november",
]


async def main() -> None:
    settings = Settings.from_env()
    planner = LLMPlanner(make_llm_client(settings))
    times = []
    slot_counts = []
    for q in QUERIES:
        t0 = time.perf_counter()
        try:
            plan = await planner.plan(q, timeout=20.0)
            dt = time.perf_counter() - t0
            times.append(dt)
            slot_counts.append(len(plan.slots))
            print(f"{dt * 1000:7.0f} ms  {len(plan.slots)} slots  {q[:44]}")
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not crash
            print(f"   FAIL  {type(exc).__name__}: {exc}  {q[:44]}")
    if not times:
        return
    times.sort()
    ms = [t * 1000 for t in times]
    within = {b: sum(1 for t in times if t <= b) / len(times) for b in (0.25, 0.35, 0.5, 1.0, 2.0)}
    print(
        json.dumps(
            {
                "n": len(ms),
                "p50_ms": round(statistics.median(ms), 1),
                "min_ms": round(ms[0], 1),
                "max_ms": round(ms[-1], 1),
                "mean_slots": round(statistics.mean(slot_counts), 2),
                "within_budget": {str(k): round(v, 3) for k, v in within.items()},
            },
            indent=2,
        )
    )


asyncio.run(main())
