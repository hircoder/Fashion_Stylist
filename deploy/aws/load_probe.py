"""Concurrency ramp against the live endpoint. Reports p50/p95/p99 and errors per
level so tail behavior is visible, not just the average.

  python3 load_probe.py https://HOST --levels 1,4,8,16 --per 24 --mode mixed

modes: repeat (same six queries over and over, cache friendly),
       unique  (every request a fresh string, cache hostile),
       mixed   (half and half, the honest one)
"""

import argparse
import concurrent.futures as cf
import http.client
import json
import ssl
import time
from urllib.parse import urlparse

EXAMPLES = [
    "I need an outfit to go to the beach this summer",
    "warm waterproof boots for hiking in the snow, under $80",
    "what should my husband wear to an outdoor wedding in june",
    "something cozy for working from home in winter",
    "a gift for my 6 year old daughter who loves unicorns",
    "smart casual outfit for a job interview at a startup",
]


def one(host: str, scheme: str, query: str) -> tuple[float, int, float]:
    """(wall seconds, status, server total_ms). Fresh connection per call."""
    ctx = ssl.create_default_context() if scheme == "https" else None
    conn = (
        http.client.HTTPSConnection(host, context=ctx, timeout=30)
        if scheme == "https"
        else http.client.HTTPConnection(host, timeout=30)
    )
    body = json.dumps({"query": query, "k": 3})
    t0 = time.perf_counter()
    try:
        conn.request("POST", "/recommend", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        payload = resp.read()
        wall = time.perf_counter() - t0
        server_ms = 0.0
        if resp.status == 200:
            try:
                server_ms = json.loads(payload)["timings"]["total_ms"]
            except Exception:  # noqa: BLE001
                pass
        return wall, resp.status, server_ms
    finally:
        conn.close()


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--per", type=int, default=24, help="requests per level")
    ap.add_argument("--mode", default="mixed", choices=["repeat", "unique", "mixed"])
    args = ap.parse_args()
    u = urlparse(args.url)
    host, scheme = u.netloc, u.scheme

    out = {}
    seq = 0
    for level in [int(x) for x in args.levels.split(",")]:
        queries = []
        for i in range(args.per):
            seq += 1
            if args.mode == "repeat" or (args.mode == "mixed" and i % 2 == 0):
                queries.append(EXAMPLES[i % len(EXAMPLES)])
            else:
                queries.append(f"outfit number {seq} for a tuesday, something plain")
        walls, statuses, servers = [], [], []
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=level) as ex:
            for wall, status, server_ms in ex.map(lambda q: one(host, scheme, q), queries):
                walls.append(wall)
                statuses.append(status)
                servers.append(server_ms)
        elapsed = time.perf_counter() - t0
        errors = {s: statuses.count(s) for s in set(statuses) if s != 200}
        row = {
            "n": len(walls),
            "rps": round(len(walls) / elapsed, 1),
            "wall_p50_ms": round(pct(walls, 0.5) * 1000, 0),
            "wall_p95_ms": round(pct(walls, 0.95) * 1000, 0),
            "wall_p99_ms": round(pct(walls, 0.99) * 1000, 0),
            "server_p50_ms": round(pct(servers, 0.5), 0),
            "server_p95_ms": round(pct(servers, 0.95), 0),
            "errors": errors,
        }
        out[f"c{level}"] = row
        print(f"c={level:<3} {json.dumps(row)}")
    print(json.dumps({"mode": args.mode, "levels": out}))


main()
