# ruff: noqa: E501  (the box texts are long on purpose)
"""Draws docs/architecture.svg. Boxes are sized from their (wrapped) text and arrows
run edge to edge, so the picture can be regenerated after a change without anything
overlapping. Then: rsvg-convert -f pdf docs/architecture.svg -o docs/architecture.pdf

    python scripts/architecture_svg.py
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "docs" / "architecture.svg"

# Helvetica advance widths (per 1000 em), enough for an honest wrap estimate
W: dict[str, int] = {}
for _ch in "abcdeghknopqsuvxyz":
    W[_ch] = 556
for _ch in "fIjlt.,:;'|!()[]/ ":
    W[_ch] = 278
for _ch in "0123456789":
    W[_ch] = 556
for _ch in "ABEKPSTVXYZ":
    W[_ch] = 667
for _ch in "CDHNRUW":
    W[_ch] = 722
W.update(
    {
        "i": 222,
        "m": 833,
        "w": 722,
        "r": 333,
        " ": 278,
        "-": 333,
        "+": 584,
        "=": 584,
        "<": 584,
        ">": 584,
        "_": 556,
        "*": 389,
        '"': 355,
        "%": 889,
        "&": 667,
        "#": 556,
        "$": 556,
        "@": 1015,
        "~": 584,
        "F": 611,
        "G": 778,
        "J": 500,
        "L": 556,
        "M": 833,
        "O": 778,
        "Q": 778,
        "I": 278,
    }
)


def width(text, size, bold=False):
    w = sum(W.get(c, 600) for c in text) / 1000 * size
    return w * (1.06 if bold else 1.0)


def wrap(text, size, max_w, bold=False):
    lines = []
    for para in text.split("\n"):
        words = para.split(" ")
        cur = ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if width(trial, size, bold) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = wd
        lines.append(cur)
    return lines


class Box:
    def __init__(self, key, title, body, x, y, w, kind="box", small=None, min_h=0):
        self.key, self.title, self.body, self.x, self.y, self.w, self.kind, self.small = (
            key,
            title,
            body,
            x,
            y,
            w,
            kind,
            small,
        )
        pad = 12
        self.tl = wrap(title, 14, w - 2 * pad, bold=True)
        self.bl = wrap(body, 12.5, w - 2 * pad)
        self.sl = wrap(small, 11, w - 2 * pad) if small else []
        self.h = max(
            min_h,
            pad
            + len(self.tl) * 18
            + len(self.bl) * 16.5
            + (len(self.sl) * 14.5 + 4 if self.sl else 0)
            + pad,
        )

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y + self.h

    def svg(self):
        fill = {"box": "#ffffff", "store": "#f4f4f4", "llm": "#ffffff", "note": "#fafafa"}[
            self.kind
        ]
        dash = ' stroke-dasharray="6 4"' if self.kind == "llm" else ""
        out = [
            f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h:.0f}" rx="4" fill="{fill}" stroke="#111" stroke-width="1.3"{dash}/>'
        ]
        y = self.y + 12 + 13
        for ln in self.tl:
            out.append(f'<text x="{self.x + 12}" y="{y:.0f}" class="h">{escape(ln)}</text>')
            y += 18
        y += 1
        for ln in self.bl:
            out.append(f'<text x="{self.x + 12}" y="{y:.0f}" class="t">{escape(ln)}</text>')
            y += 16.5
        if self.sl:
            y += 3
            for ln in self.sl:
                out.append(f'<text x="{self.x + 12}" y="{y:.0f}" class="s">{escape(ln)}</text>')
                y += 14.5
        return "\n".join(out)


MONO = 6.6  # px per character at 11px Menlo


def label(x, y, text, anchor="middle", cls="lbl"):
    w = len(text) * MONO + 8
    bx = x - (w / 2 if anchor == "middle" else (w if anchor == "end" else 0))
    return (
        f'<rect x="{bx:.0f}" y="{y - 10:.0f}" width="{w:.0f}" height="14" fill="#fff" opacity="0.95"/>'
        f'<text x="{x:.0f}" y="{y:.0f}" class="{cls}" text-anchor="{anchor}">{escape(text)}</text>'
    )


def gap_label(xc, y_arrow, gap_w, text, below=False):
    """Label wrapped to the width of a gap between two boxes, stacked above (or below) the arrow."""
    max_chars = max(4, int((gap_w - 6) / MONO))
    lines, cur = [], ""
    for wd in text.split():
        trial = (cur + " " + wd).strip()
        if len(trial) <= max_chars or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = wd
    lines.append(cur)
    out = []
    for i, ln in enumerate(lines):
        y = (y_arrow + 14 + i * 13) if below else (y_arrow - 6 - (len(lines) - 1 - i) * 13)
        out.append(label(xc, y, ln))
    return "\n".join(out)


def arrow(points, cls="line", both=False):
    d = "M " + " L ".join(f"{x:.0f} {y:.0f}" for x, y in points)
    extra = ' marker-start="url(#arr)"' if both else ""
    return f'<path d="{d}" class="{cls}"{extra}/>'


# ------------------------------------------------------------------ layout
WIDTH = 1720
M = 40  # margin
COLS = 6
GAP = 66
BW = (WIDTH - 2 * M - (COLS - 1) * GAP) / COLS  # ~ 236


def colx(i):
    return M + i * (BW + GAP)


svg = []
y = 0
svg.append(
    f'<text x="{M}" y="44" class="title">Fashion stylist: how the service is built and how a request moves through it</text>'
)
svg.append(
    f'<text x="{M}" y="66" class="sub">POST /recommend takes a sentence like "an outfit for the beach this summer under $150", plans 1 to 5 product slots with an LLM, retrieves per slot with dense + BM25 search over 100,000 indexed listings, reranks each slot with an LLM, and returns k products per slot with reasons. Solid arrows carry data, dashed arrows are LLM calls, grey boxes are files.</text>'
)

# ---- offline lane
lane1_y = 92
off = []
off.append(
    Box(
        "cmd",
        "Commands (Makefile)",
        "make data: download the 224 MB gzip from the McAuley lab mirror\nmake ingest: stylist ingest\nmake index: stylist build-index --limit 100000 --sampling popular\nmake demo: 486 row fixture + hash embedder, no model download",
        colx(0),
        lane1_y + 46,
        BW,
        kind="note",
    )
)
off.append(
    Box(
        "raw",
        "meta_Amazon_Fashion.jsonl.gz (raw)",
        "826,108 listings, 224 MB gzip, one json object per line: title, price, average_rating, rating_number, store, features, description, details, images, parent_asin",
        colx(1),
        lane1_y + 46,
        BW,
        kind="store",
        small="price known for 6% of rows; categories empty for every row, so no taxonomy",
    )
)
off.append(
    Box(
        "ingest",
        "stylist ingest (catalog.py)",
        'streams the file, one typed row per listing\nparse_price: floats and "$12.99" kept, ranges and junk -> null + status\nderive_audience: details.Department, else title words\ndoc_text = title | features | department, material, style, colour, brand | store',
        colx(2),
        lane1_y + 46,
        BW,
        small="bad lines skipped and counted, written to a temp file then renamed; ~30 s",
    )
)
off.append(
    Box(
        "parq",
        "catalog.parquet (data/processed/)",
        "826,108 rows, zstd, 150 MB\nrow_id, parent_asin, title, price, price_status, audience, average_rating (null when missing), rating_number, store, image_url, doc_text, details",
        colx(3),
        lane1_y + 46,
        BW,
        kind="store",
    )
)
off.append(
    Box(
        "build",
        "stylist build-index (index.py)",
        "pick rows: popular 100K (default), random, or all (Arrow side, nothing else materialised)\nembed doc_text with BAAI/bge-small-en-v1.5: 384-d, L2 normalised, batches of 128\nBM25 (bm25s) over the same text\nsha256 of every file into meta.json",
        colx(4),
        lane1_y + 46,
        BW,
        small="2.7 min for 100K rows on a laptop GPU; built in a scratch dir, swapped in at the end",
    )
)
off.append(
    Box(
        "idx",
        "data/index/ (ships as one tarball)",
        "embeddings.npy float16 (100000 x 384)\nrow_ids.npy int64\ncatalog.parquet: the indexed rows, serving columns\nbm25/\nmeta.json: pipeline version, model + revision, dim, counts, sampling, checksums",
        colx(5),
        lane1_y + 46,
        BW,
        kind="store",
        small="Railway: INDEX_URL + INDEX_SHA256, verifed and unpacked at boot",
    )
)
h1 = max(b.h for b in off)
for b in off:
    b.h = h1
lane1_h = 46 + h1 + 40
svg.append(
    f'<rect x="{M - 14}" y="{lane1_y}" width="{WIDTH - 2 * M + 28}" height="{lane1_h:.0f}" rx="8" fill="none" stroke="#bbb"/>'
)
svg.append(
    f'<text x="{M}" y="{lane1_y + 28}" class="lane">1. Offline, run once per catalog: raw json to a verified index directory</text>'
)
for b in off:
    svg.append(b.svg())


def h_arrow(a, b, text, dy=-6):
    svg.append(arrow([(a.right, a.cy), (b.x - 1, b.cy)]))
    svg.append(gap_label((a.right + b.x) / 2, a.cy, b.x - a.right, text))


h_arrow(off[1], off[2], "json lines")
h_arrow(off[2], off[3], "typed rows")
h_arrow(off[3], off[4], "selected rows")
h_arrow(off[4], off[5], "artifacts")
svg.append(
    f'<text x="{M}" y="{lane1_y + lane1_h - 14}" class="s">Startup check (SearchIndex.load): pipeline version, embedding model name and revision, row counts, row order (row_ids vs catalog), sha256 of every file, finite embeddings. Anything off and the service refuses to start rather than serve misaligned rows.</text>'
)

# ---- online lane
lane2_y = lane1_y + lane1_h + 40
r1 = lane2_y + 46
on = {}
on["client"] = Box(
    "client",
    "Client",
    "React page (ui/), curl, or any HTTP client\nPOST /recommend {query, k, max_price, min_price, audience, include_unpriced, use_llm, rerank}\nOpenAPI at /docs, health at /health, readiness at /ready",
    colx(0),
    r1,
    BW,
)
on["api"] = Box(
    "api",
    "FastAPI (api.py)",
    "pydantic validation: unknown fields rejected, query 1 to 500 chars, k 1 to 10, finite prices, min <= max\nrequest id, per-ip rate limit, in-flight cap, body size limit\none 40 s deadline per request; 422 / 429 / 503 / 504 error bodies",
    colx(1),
    r1,
    BW,
)
on["plan"] = Box(
    "plan",
    "1. Planner (planner.py)",
    "LLM structured output -> PlannerOutput: intent, audience, occasion, season, budget + scope (per item / total), style words, 1 to 5 slots: name, listing-style search query, 2 to 6 type keywords, up to 4 exclude words, budget share\nnormalize_plan: caps, dedupe, 10% floor per slot, never above the total",
    colx(2),
    r1,
    BW,
    small="cache per (query, mode, provider, model, prompt version); identical concurrent requests share one call; regex planner if the LLM fails or there is no key",
)
on["merge"] = Box(
    "merge",
    "2. Merge constraints",
    "request fields override the plan\none SlotWindow per slot: min / max price, audience, unpriced policy (strict for an explicit bound, flagged backfill for an inferred one)",
    colx(3),
    r1,
    BW,
)
on["retr"] = Box(
    "retr",
    "3. Retriever (retrieval.py)",
    "all slot queries embedded in one batch, ONE matmul Q (5 x 384) against E (384 x 100K)\nBM25 score for every row\naudience + price masks applied before top-100\nreciprocal rank fusion (k = 60), + keyword boost, - exclude penalty, + Bayesian rating prior, + audience match\none row per variant group (group_key from the title)",
    colx(4),
    r1,
    BW,
    small="50 to 300 ms; unpriced pool ranked alongside and flagged when allowed",
)
on["mem"] = Box(
    "mem",
    "In memory (loaded once)",
    "SearchIndex: embeddings float32 (100K x 384), bm25s index, serving catalog (pandas), row_ids\nembedding model bge-small-en-v1.5 (query prefix), CPU or GPU\ncolumn caches for the masks, group keys memoised",
    colx(5),
    r1,
    BW,
    kind="store",
)
hh = max(b.h for b in on.values())
for b in on.values():
    b.h = hh
r2 = r1 + hh + 86
on["llm"] = Box(
    "llm",
    "LLM provider (llm/)",
    "Anthropic or OpenAI SDK behind one async method complete_json(system, user, schema)\nstructured output parsed into pydantic models; typed errors: auth, rate limit, timeout, refusal, truncation, validation\nglobal concurrency cap (LLM_CONCURRENCY = 8), per stage budgets: plan 15 s, rerank 20 s",
    colx(0),
    r2,
    BW * 2 + GAP,
    kind="llm",
    small="LLM_PROVIDER=none runs the whole service without a key: regex plan + hybrid retrieval + deterministic reasons",
)
on["resp"] = Box(
    "resp",
    "6. Response (schemas.py)",
    "request_id, plan as used, slots[]: name, search_query, keywords, exclude_keywords, budget_max, n_eligible, items[]: rank, parent_asin, title, price, price_known, average_rating, rating_number, store, audience, image_url, url, score, matched_keywords, reason, evidence\nnote, warnings[], index_info, llm_info, timings",
    colx(2),
    r2,
    BW,
)
on["sel"] = Box(
    "sel",
    "5. Select (service.py)",
    "round robin across slots by rank\na product group fills at most one slot\ntop k per slot\nwarnings for every fallback: slot below k, unpriced items under a budget, priced picks over a stated total, planner or rerank fallback",
    colx(3),
    r2,
    BW,
)
on["rerank"] = Box(
    "rerank",
    "4. Reranker (reranker.py)",
    "one LLM call per slot, all slots in parallel (asyncio.wait, rerank budget)\nprompt: request, plan summary, slot, top 10 candidates as compact json labelled untrusted\noutput: picks[] (row_id, reason <= 15 words, evidence), no_good_match, note\nids validated (offered, unique, <= k); a failed or late slot keeps retrieval order; urls stripped from prose",
    colx(4),
    r2,
    BW,
    small="3 to 7 s with claude-sonnet-4-6",
)
on["notes"] = Box(
    "notes",
    "Timing and fallbacks (laptop M4, claude-sonnet-4-6)",
    "plan 3 to 7 s (cached: 0 ms), retrieve 0.05 to 0.3 s, rerank 3 to 7 s, total 6 to 12 s\nuse_llm=false: regex plan + retrieval only, < 1 s\nany LLM failure degrades one stage, never the request; every fallback is a warning in the response",
    colx(5),
    r2,
    BW,
    kind="note",
)
h2 = max(on[k].h for k in ("llm", "resp", "sel", "rerank", "notes"))
for k in ("llm", "resp", "sel", "rerank", "notes"):
    on[k].h = h2
lane2_h = 46 + hh + 86 + h2 + 70
svg.append(
    f'<rect x="{M - 14}" y="{lane2_y}" width="{WIDTH - 2 * M + 28}" height="{lane2_h:.0f}" rx="8" fill="none" stroke="#bbb"/>'
)
svg.append(
    f'<text x="{M}" y="{lane2_y + 28}" class="lane">2. Online, one request: plan, retrieve, rerank, select (one deadline, every fallback reported)</text>'
)
for b in on.values():
    svg.append(b.svg())

# row 1 arrows
h_arrow(on["client"], on["api"], "POST json")
h_arrow(on["api"], on["plan"], "query + fields")
h_arrow(on["plan"], on["merge"], "plan (1-5 slots)")
h_arrow(on["merge"], on["retr"], "slot windows")
# retriever <-> memory: two arrows
a, b = on["retr"], on["mem"]
svg.append(arrow([(a.right, a.cy - 16), (b.x - 1, a.cy - 16)]))
svg.append(gap_label((a.right + b.x) / 2, a.cy - 16, b.x - a.right, "5 query vectors, tokens"))
svg.append(arrow([(b.x, a.cy + 16), (a.right + 1, a.cy + 16)]))
svg.append(
    gap_label((a.right + b.x) / 2, a.cy + 16, b.x - a.right, "scores per row, rows", below=True)
)
# index dir -> memory (offline to online)
svg.append(arrow([(off[5].cx, off[5].bottom), (on["mem"].cx, on["mem"].y - 1)]))
svg.append(
    label(
        off[5].cx - 8,
        (off[5].bottom + on["mem"].y) / 2 + 4,
        "loaded + verified at startup",
        anchor="end",
    )
)
# retriever -> reranker (down)
svg.append(arrow([(on["retr"].cx, on["retr"].bottom), (on["rerank"].cx, on["rerank"].y - 1)]))
svg.append(
    label(
        on["retr"].cx - 8,
        (on["retr"].bottom + on["rerank"].y) / 2 + 4,
        "candidate pools: max(10, k x slots) per slot, grouped, masked",
        anchor="end",
    )
)


# row 2 arrows (leftwards)
def h_arrow_left(a, b, text, dy=-6):
    svg.append(arrow([(a.x, a.cy), (b.right + 1, b.cy)]))
    svg.append(gap_label((a.x + b.right) / 2, a.cy, a.x - b.right, text))


h_arrow_left(on["rerank"], on["sel"], "ranked + reasons")
h_arrow_left(on["sel"], on["resp"], "k per slot")
# response -> client: left to the client column, then up
cl, rs = on["client"], on["resp"]
lane_y = rs.y - 24
svg.append(arrow([(rs.cx, rs.y), (rs.cx, lane_y), (cl.cx, lane_y), (cl.cx, cl.bottom + 1)]))
svg.append(
    label(cl.cx + 14, lane_y - 6, "200 json (slots, items, reasons, warnings)", anchor="start")
)
# planner -> llm (dashed, both ways): planner bottom-left down to the lane, left, down into llm top
pl, lm = on["plan"], on["llm"]
px = pl.x + 40
ly2 = pl.bottom + 22
svg.append(
    arrow(
        [(px, pl.bottom), (px, ly2), (lm.right - 60, ly2), (lm.right - 60, lm.y - 1)],
        cls="dashed",
        both=True,
    )
)
svg.append(label(lm.right - 60 - 8, ly2 - 6, "prompt + json schema  <->  plan json", anchor="end"))
# reranker -> llm (dashed, both ways): reranker bottom down to a lane under row 2, left, up into llm bottom
rr = on["rerank"]
ly3 = rr.bottom + 34
svg.append(
    arrow(
        [(rr.cx, rr.bottom), (rr.cx, ly3), (lm.cx, ly3), (lm.cx, lm.bottom + 1)],
        cls="dashed",
        both=True,
    )
)
svg.append(
    label(
        (rr.cx + lm.cx) / 2,
        ly3 - 6,
        "5 prompts, 10 candidates each  <->  picks + reasons (one call per slot, in parallel)",
    )
)

HEIGHT = int(lane2_y + lane2_h + 30)
head = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#111"/></marker>
    <style>
      .title {{ font-size: 24px; font-weight: bold; fill: #111; }}
      .sub {{ font-size: 12.5px; fill: #444; }}
      .lane {{ font-size: 17px; font-weight: bold; fill: #111; }}
      .h {{ font-size: 14px; font-weight: bold; fill: #111; }}
      .t {{ font-size: 12.5px; fill: #222; }}
      .s {{ font-size: 11px; fill: #666; }}
      .lbl {{ font-size: 11px; fill: #111; font-family: Menlo, Consolas, monospace; }}
      .line {{ stroke: #111; stroke-width: 1.4; fill: none; marker-end: url(#arr); }}
      .dashed {{ stroke: #111; stroke-width: 1.4; fill: none; stroke-dasharray: 6 4; marker-end: url(#arr); }}
    </style>
  </defs>
  <rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#fff"/>
'''
# subtitle is long: wrap it into two lines
sub_lines = wrap(
    'POST /recommend takes a sentence like "an outfit for the beach this summer under $150", plans 1 to 5 product slots with an LLM, retrieves per slot with dense + BM25 search over 100,000 indexed listings, reranks each slot with an LLM, and returns k products per slot with reasons. Solid arrows carry data, dashed arrows are LLM calls (both directions), grey boxes are files on disk, the dotted box is the only external service.',
    12.5,
    WIDTH - 2 * M,
)
svg[1] = "\n".join(
    f'<text x="{M}" y="{62 + i * 16}" class="sub">{escape(line)}</text>'
    for i, line in enumerate(sub_lines)
)
body = "\n".join(svg)
open(OUT, "w").write(head + body + "\n</svg>\n")
print("svg", WIDTH, HEIGHT)
