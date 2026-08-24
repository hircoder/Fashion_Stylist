"""Acceptance battery for the live AWS deployment. Sections are picked by name so the
orchestration can interleave SSM work (limiter off, restarts) between them.

  python3 test_battery.py https://HOST contract endpoints guardrail
  python3 test_battery.py https://HOST steady fresh ramp soak cache consistency

Every section prints one json line and the driver collects them into the experiment
files. Client-side only, stdlib only.
"""

import concurrent.futures as cf
import http.client
import json
import ssl
import sys
import threading
import time
from urllib.parse import urlparse

BASE = sys.argv[1].rstrip("/")
U = urlparse(BASE)
HOST, SCHEME = U.netloc, U.scheme
WARM_QUERY = "I need an outfit to go to the beach this summer"
EXAMPLES = [
    WARM_QUERY,
    "warm waterproof boots for hiking in the snow, under $80",
    "what should my husband wear to an outdoor wedding in june",
    "something cozy for working from home in winter",
    "a gift for my 6 year old daughter who loves unicorns",
    "smart casual outfit for a job interview at a startup",
]
_seq_lock = threading.Lock()
_seq = [0]


def unique_query() -> str:
    with _seq_lock:
        _seq[0] += 1
        n = _seq[0]
    return f"outfit number {n} for a plain tuesday, nothing fancy"


def conn():
    if SCHEME == "https":
        return http.client.HTTPSConnection(HOST, context=ssl.create_default_context(), timeout=35)
    return http.client.HTTPConnection(HOST, timeout=35)


def post(c, body: dict, path="/recommend", headers=None):
    payload = json.dumps(body)
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    t0 = time.perf_counter()
    c.request("POST", path, payload, h)
    r = c.getresponse()
    data = r.read()
    return time.perf_counter() - t0, r.status, data, dict(r.getheaders())


def one_shot(body: dict):
    c = conn()
    try:
        return post(c, body)
    finally:
        c.close()


def pct(vals, p):
    if not vals:
        return 0.0
    v = sorted(vals)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def stats(walls, servers=None, statuses=None):
    out = {
        "n": len(walls),
        "p50_ms": round(pct(walls, 0.5) * 1000, 1),
        "p90_ms": round(pct(walls, 0.9) * 1000, 1),
        "p95_ms": round(pct(walls, 0.95) * 1000, 1),
        "p99_ms": round(pct(walls, 0.99) * 1000, 1),
        "max_ms": round(max(walls) * 1000, 1) if walls else 0,
    }
    if servers is not None:
        out["server_p50_ms"] = round(pct(servers, 0.5), 1)
        out["server_p95_ms"] = round(pct(servers, 0.95), 1)
    if statuses is not None:
        out["errors"] = {str(s): statuses.count(s) for s in set(statuses) if s != 200}
    return out


def emit(section, payload):
    print(json.dumps({"section": section, **payload}, default=str), flush=True)


# ---------------------------------------------------------------- sections


def sec_steady():
    """One keep-alive connection, warm query, n=100: the floor of the deployment."""
    c = conn()
    try:
        for _ in range(3):
            post(c, {"query": WARM_QUERY, "k": 3})
        walls, servers = [], []
        for _ in range(100):
            w, s, data, _ = post(c, {"query": WARM_QUERY, "k": 3})
            walls.append(w)
            if s == 200:
                servers.append(json.loads(data)["timings"]["total_ms"])
    finally:
        c.close()
    emit("steady_keepalive_n100", stats(walls, servers))


def sec_fresh():
    """New TLS connection per request, n=50: what a first click pays."""
    walls, servers, statuses = [], [], []
    for _ in range(50):
        w, s, data, _ = one_shot({"query": WARM_QUERY, "k": 3})
        walls.append(w)
        statuses.append(s)
        if s == 200:
            servers.append(json.loads(data)["timings"]["total_ms"])
    emit("fresh_tls_n50", stats(walls, servers, statuses))


def sec_endpoints():
    """Every surface, plus the edge cache on /assets (second hit must come from CloudFront)."""
    rows = {}
    c = conn()
    try:
        for path in ("/", "/overview", "/health", "/ready", "/docs"):
            t0 = time.perf_counter()
            c.request("GET", path)
            r = c.getresponse()
            body = r.read()
            rows[path] = {"status": r.status, "ms": round((time.perf_counter() - t0) * 1000, 1)}
            if path == "/":
                import re

                m = re.search(rb'src="(/assets/[^"]+\.js)"', body)
                rows["_asset_path"] = m.group(1).decode() if m else None
        asset = rows.pop("_asset_path", None)
        if asset:
            for label in ("asset_first", "asset_second"):
                t0 = time.perf_counter()
                c.request("GET", asset)
                r = c.getresponse()
                r.read()
                rows[label] = {
                    "status": r.status,
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                    "x_cache": dict(r.getheaders()).get("X-Cache", ""),
                }
        c.request("GET", "/")
        r = c.getresponse()
        r.read()
        hd = {k.lower(): v for k, v in r.getheaders()}  # cloudfront lowercases names
        rows["headers"] = {
            "alt_svc_h3": "h3" in hd.get("alt-svc", ""),
            "csp": "content-security-policy" in hd,
            "nosniff": hd.get("x-content-type-options", "") == "nosniff",
            "frame_deny": hd.get("x-frame-options", "") == "DENY",
            "via_cloudfront": "cloudfront" in hd.get("via", "").lower(),
        }
    finally:
        c.close()
    emit("endpoints", rows)


def sec_contract():
    """The api must refuse malformed input with the documented status and body shape."""
    cases = []

    def case(name, status_want, body, path="/recommend", method="POST", raw=None, headers=None):
        c = conn()
        try:
            payload = raw if raw is not None else json.dumps(body)
            h = {"Content-Type": "application/json"}
            if headers:
                h.update(headers)
            c.request(method, path, payload, h)
            r = c.getresponse()
            data = r.read()
            ok = r.status == status_want
            detail = ""
            if not ok:
                detail = f"got {r.status}"
            cases.append({"case": name, "want": status_want, "ok": ok, "detail": detail})
            return data
        finally:
            c.close()

    case("valid_minimal", 200, {"query": "boots", "k": 1})
    case("empty_query", 422, {"query": "", "k": 3})
    case("query_too_long", 422, {"query": "x" * 501, "k": 3})
    case("k_zero", 422, {"query": "boots", "k": 0})
    case("k_eleven", 422, {"query": "boots", "k": 11})
    case("unknown_field", 422, {"query": "boots", "k": 3, "surprise": 1})
    case("negative_price", 422, {"query": "boots", "k": 3, "max_price": -5})
    case("min_over_max", 422, {"query": "boots", "k": 3, "min_price": 90, "max_price": 10})
    case("bad_json", 422, None, raw="{not json")
    case("oversized_body", 413, {"query": "boots", "k": 3, "pad": "x" * 20000})
    case("wrong_method", 405, None, method="GET", raw="")
    data = case("full_valid", 200, {"query": "warm boots under $60", "k": 2, "rerank": False})
    r = json.loads(data)
    for field in (
        "request_id",
        "plan",
        "slots",
        "warnings",
        "index_info",
        "llm_info",
        "timings",
        "served_from_cache",
    ):
        cases.append({"case": f"field_{field}", "want": "present", "ok": field in r, "detail": ""})
    passed = sum(1 for x in cases if x["ok"])
    emit(
        "contract",
        {
            "passed": passed,
            "failed": len(cases) - passed,
            "cases": [c for c in cases if not c["ok"]],
        },
    )


def sec_guardrail():
    """Blow through the rate limit on purpose; the envelope should be fast 429s with a
    retry-after style body, then a clean recovery."""
    c = conn()
    statuses, walls_429 = [], []
    body_shape_ok = True
    try:
        for _ in range(70):
            w, s, data, _ = post(c, {"query": WARM_QUERY, "k": 3})
            statuses.append(s)
            if s == 429:
                walls_429.append(w)
                try:
                    err = json.loads(data)["error"]
                    body_shape_ok = body_shape_ok and ("code" in err and "message" in err)
                except Exception:  # noqa: BLE001
                    body_shape_ok = False
    finally:
        c.close()
    first_429 = statuses.index(429) + 1 if 429 in statuses else None
    time.sleep(20)
    _, s_after, _, _ = one_shot({"query": WARM_QUERY, "k": 3})
    emit(
        "guardrail_rate_limit",
        {
            "sent": len(statuses),
            "ok": statuses.count(200),
            "throttled": statuses.count(429),
            "first_429_at_request": first_429,
            "throttle_p50_ms": round(pct(walls_429, 0.5) * 1000, 1) if walls_429 else None,
            "error_body_shape_ok": body_shape_ok,
            "recovered_status_after_20s": s_after,
        },
    )


def _ramp_level(level, per, mode):
    def q(i):
        if mode == "repeat" or (mode == "mixed" and i % 2 == 0):
            return EXAMPLES[i % len(EXAMPLES)]
        return unique_query()

    bodies = [{"query": q(i), "k": 3} for i in range(per)]
    walls, statuses, servers = [], [], []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=level) as ex:
        for w, s, data, _ in ex.map(one_shot, bodies):
            walls.append(w)
            statuses.append(s)
            if s == 200:
                servers.append(json.loads(data)["timings"]["total_ms"])
    dt = time.perf_counter() - t0
    row = stats(walls, servers, statuses)
    row["rps"] = round(len(walls) / dt, 1)
    return row


def sec_ramp():
    out = {}
    for mode in ("repeat", "unique", "mixed"):
        for level in (1, 2, 4, 8, 16, 24, 32):
            out[f"{mode}_c{level}"] = _ramp_level(level, max(32, level * 3), mode)
            emit("ramp_progress", {"done": f"{mode}_c{level}", **out[f"{mode}_c{level}"]})
    emit("ramp", out)


def sec_soak():
    """Five minutes at c=6, mixed traffic, per-minute percentiles: drift is the point."""
    stop_at = time.time() + 300
    lock = threading.Lock()
    rows = []  # (t, wall, status)

    def worker():
        i = 0
        while time.time() < stop_at:
            body = {"query": EXAMPLES[i % 6] if i % 2 == 0 else unique_query(), "k": 3}
            try:
                w, s, _, _ = one_shot(body)
            except Exception:  # noqa: BLE001
                w, s = 0.0, -1
            with lock:
                rows.append((time.time(), w, s))
            i += 1

    threads = [threading.Thread(target=worker) for _ in range(6)]
    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    minutes = {}
    for ts, w, s in rows:
        m = int((ts - t_start) // 60)
        minutes.setdefault(m, {"walls": [], "statuses": []})
        minutes[m]["walls"].append(w)
        minutes[m]["statuses"].append(s)
    per_min = {
        f"min{m}": {
            "n": len(v["walls"]),
            "p50_ms": round(pct(v["walls"], 0.5) * 1000, 1),
            "p95_ms": round(pct(v["walls"], 0.95) * 1000, 1),
            "errors": sum(1 for s in v["statuses"] if s != 200),
        }
        for m, v in sorted(minutes.items())
    }
    walls = [w for _, w, _ in rows]
    statuses = [s for _, _, s in rows]
    total = stats(walls, statuses=statuses)
    total["rps"] = round(len(rows) / 300, 1)
    emit("soak_5min_c6", {"total": total, "per_minute": per_min})


def sec_restart():
    """A minute of c=4 repeat traffic; the driver restarts the service at ~t+15 s.
    Measures the error window a deploy costs."""
    stop_at = time.time() + 60
    lock = threading.Lock()
    rows = []

    def worker():
        i = 0
        while time.time() < stop_at:
            try:
                w, s, _, _ = one_shot({"query": EXAMPLES[i % 6], "k": 3})
            except Exception:  # noqa: BLE001
                w, s = 0.0, -1
            with lock:
                rows.append((time.time(), w, s))
            i += 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bad = [(ts - t0, s) for ts, _, s in rows if s != 200]
    window = (max(b[0] for b in bad) - min(b[0] for b in bad)) if bad else 0.0
    emit(
        "restart_under_traffic",
        {
            "n": len(rows),
            "errors": len(bad),
            "error_statuses": sorted({s for _, s in bad}),
            "error_window_s": round(window, 1),
            "first_error_at_s": round(min((b[0] for b in bad), default=0), 1),
            "recovered": rows[-1][2] == 200 if rows else False,
        },
    )


def sec_cache():
    """The ladder, with assertions this time. One keep-alive connection for the whole
    ladder, like a browser session: a fresh connection per rung can land on either
    worker and each worker warms its own caches."""
    q = unique_query() + " with a scarf"
    para = q.replace("nothing fancy with a scarf", "and nothing fancy plus a scarf")
    c = conn()
    try:
        w1, s1, d1, _ = post(c, {"query": q, "k": 3})
        r1 = json.loads(d1)
        time.sleep(6)
        w2, s2, d2, _ = post(c, {"query": q, "k": 3})
        r2 = json.loads(d2)
        w3, s3, d3, _ = post(c, {"query": q, "k": 3})
        r3 = json.loads(d3)
        w4, s4, d4, _ = post(c, {"query": para, "k": 3})
        r4 = json.loads(d4)
    finally:
        c.close()
    emit(
        "cache_ladder",
        {
            "cold": {"ms": round(w1 * 1000), "planner": r1["llm_info"]["planner_used"]},
            "after_background": {
                "ms": round(w2 * 1000),
                "planner": r2["llm_info"]["planner_used"],
                "plan_cache_hit": r2["llm_info"]["plan_cache_hit"],
            },
            "response_cache": {
                "ms": round(w3 * 1000),
                "served_from_cache": r3["served_from_cache"],
                "fresh_request_id": r3["request_id"] != r2["request_id"],
                "zero_llm_calls_reported": r3["llm_info"]["calls"] == 0,
            },
            "paraphrase_semantic": {
                "ms": round(w4 * 1000),
                "planner": r4["llm_info"]["planner_used"],
                "new_llm_calls": r4["llm_info"]["calls"],
            },
            "pass": (
                r2["llm_info"]["planner_used"] == "llm"
                and r3["served_from_cache"] is True
                and r4["llm_info"]["planner_used"] == "llm"
                and r4["llm_info"]["calls"] == 0
            ),
        },
    )


def sec_consistency():
    """Same warm request five times: identical items every time."""
    picks = []
    for _ in range(5):
        _, s, data, _ = one_shot({"query": WARM_QUERY, "k": 3})
        r = json.loads(data)
        picks.append(tuple(i["parent_asin"] for sl in r["slots"] for i in sl["items"]))
    emit(
        "consistency_x5",
        {"identical": len(set(picks)) == 1, "items_per_answer": len(picks[0]) if picks else 0},
    )


SECTIONS = {
    "steady": sec_steady,
    "fresh": sec_fresh,
    "endpoints": sec_endpoints,
    "contract": sec_contract,
    "guardrail": sec_guardrail,
    "ramp": sec_ramp,
    "soak": sec_soak,
    "restart": sec_restart,
    "cache": sec_cache,
    "consistency": sec_consistency,
}

for name in sys.argv[2:]:
    SECTIONS[name]()
