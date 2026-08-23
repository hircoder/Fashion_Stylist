"""Command line entry points: download-data, ingest, build-index, recommend, serve."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path

from pydantic import ValidationError

from stylist.artifacts import ArtifactError
from stylist.config import ConfigError, Settings, configure_logging
from stylist.index import IndexValidationError
from stylist.service import RequestTimeout

RAW_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/"
    "meta_Amazon_Fashion.jsonl.gz"
)
# sha256 of the file as downloaded on 2026-08-22 (224,299,124 bytes). The mirror publishes
# no checksum of its own, so a mismatch is a warning, not an error: the upstream file may
# legitimately be refreshed.
RAW_SHA256 = "0b121c7494b0216ba3bf80adce9c79286fe08f14086966f841fe6716e1a24b73"
RAW_MAX_BYTES = 2 * 1024**3  # the file is 224 MB; anything near 2 GB is not it


class DownloadError(RuntimeError):
    pass


def _download(url: str, out: Path, max_bytes: int = RAW_MAX_BYTES) -> str:
    """Stream `url` to `out` through a temp file; returns the sha256 of what was written."""
    import hashlib

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    h = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or 0)
            if total > max_bytes:
                raise DownloadError(f"{url} is {total} bytes, above the {max_bytes} bytes cap")
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                done += len(chunk)
                if done > max_bytes:
                    raise DownloadError(f"download exceeded the {max_bytes} bytes cap")
                f.write(chunk)
                h.update(chunk)
                if total:
                    sys.stderr.write(f"\r{done / 1e6:8.1f} / {total / 1e6:.1f} MB")
        sys.stderr.write("\n")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return h.hexdigest()


def cmd_download(args, settings: Settings) -> int:
    out = Path(args.out or settings.raw_path)
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to re-download)")
        return 0
    print(f"downloading {RAW_URL} -> {out}")
    got = _download(RAW_URL, out)
    if got != RAW_SHA256:
        print(
            f"note: sha256 {got[:16]}... differs from the file this code was built against "
            f"({RAW_SHA256[:16]}...); the upstream file may have been refreshed",
            file=sys.stderr,
        )
    else:
        print("sha256 matches the file this code was built against")
    return 0


def cmd_ingest(args, settings: Settings) -> int:
    from stylist.catalog import ingest

    raw = Path(args.raw or settings.raw_path)
    out = Path(args.out or settings.processed_path)
    if not raw.exists():
        print(f"raw file not found: {raw} (run `stylist download-data` first)", file=sys.stderr)
        return 2
    stats = ingest(raw, out, limit=args.limit)
    print(json.dumps(stats.as_dict(), indent=2))
    print(f"wrote {stats.rows} rows -> {out}", file=sys.stderr)
    return 0


def cmd_build_index(args, settings: Settings) -> int:
    from stylist.embeddings import make_embedder
    from stylist.index import build_index

    catalog = Path(args.catalog or settings.processed_path)
    index_dir = Path(args.index_dir or settings.index_dir)
    if not catalog.exists():
        print(f"catalog not found: {catalog} (run `stylist ingest` first)", file=sys.stderr)
        return 2
    limit = None if args.sampling == "all" else args.limit
    embedder = make_embedder(settings)
    meta = build_index(
        catalog,
        index_dir,
        embedder,
        limit=limit,
        sampling=args.sampling,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(meta.to_json())
    return 0


def _format_pretty(res) -> str:
    lines = [f"query: {res.query}", f"plan ({res.plan.source}): {res.plan.intent}"]
    if res.plan.audience or res.plan.budget_max:
        lines.append(
            f"  audience={res.plan.audience} budget_max={res.plan.budget_max} "
            f"scope={res.plan.budget_scope}"
        )
    for slot in res.slots:
        lines.append("")
        lines.append(f"[{slot.name}]  search: {slot.search_query}")
        if not slot.items:
            lines.append("  (nothing found)")
        for it in slot.items:
            price = f"${it.price:.2f}" if it.price is not None else "price n/a"
            lines.append(
                f"  {it.rank}. {it.title[:90]}\n"
                f"     {price} | {it.average_rating:.1f} stars ({it.rating_number:,}) | {it.url}\n"
                f"     {it.reason}"
            )
    if res.note:
        lines += ["", f"note: {res.note}"]
    if res.warnings:
        lines += ["", "warnings:"] + [f"  - {w}" for w in res.warnings]
    lines.append("")
    lines.append(
        f"timings: {res.timings}  (planner={res.llm_info.planner_used}, "
        f"rerank={res.llm_info.rerank_used}, index rows={res.index_info.rows})"
    )
    return "\n".join(lines)


def cmd_recommend(args, settings: Settings) -> int:
    from stylist.api import build_service
    from stylist.schemas import RecommendRequest

    try:
        svc = build_service(settings)
    except IndexValidationError as exc:
        print(f"cannot load index: {exc}", file=sys.stderr)
        return 2
    req = RecommendRequest(
        query=args.query,
        k=args.k,
        max_price=args.max_price,
        min_price=args.min_price,
        audience=args.audience,
        include_unpriced=args.include_unpriced,
        use_llm=not args.no_llm,
        rerank=not args.no_rerank,
    )
    res = asyncio.run(svc.recommend(req))
    if args.json:
        print(res.model_dump_json(indent=2))
    else:
        print(_format_pretty(res))
    return 0


def cmd_serve(args, settings: Settings) -> int:
    import uvicorn

    uvicorn.run(
        "stylist.api:get_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stylist", description="semantic fashion recommendations")
    sub = p.add_subparsers(dest="command")

    d = sub.add_parser("download-data", help="download the raw Amazon Fashion metadata (224 MB)")
    d.add_argument("--out")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download)

    i = sub.add_parser("ingest", help="raw jsonl.gz -> clean catalog parquet")
    i.add_argument("--raw")
    i.add_argument("--out")
    i.add_argument("--limit", type=int, default=None, help="only read the first N rows")
    i.set_defaults(func=cmd_ingest)

    b = sub.add_parser("build-index", help="embed + bm25 index a subset of the catalog")
    b.add_argument("--catalog")
    b.add_argument("--index-dir")
    b.add_argument("--limit", type=int, default=100_000)
    b.add_argument("--sampling", choices=["popular", "random", "all"], default="popular")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--batch-size", type=int, default=128)
    b.set_defaults(func=cmd_build_index)

    r = sub.add_parser("recommend", help="run one query")
    r.add_argument("query")
    r.add_argument("--k", type=int, default=4)
    r.add_argument("--max-price", type=float)
    r.add_argument("--min-price", type=float)
    r.add_argument("--audience", choices=["women", "men", "girls", "boys", "baby", "unisex"])
    r.add_argument(
        "--include-unpriced",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow items with unknown price when a budget applies (default: auto, see README)",
    )
    r.add_argument("--no-llm", action="store_true")
    r.add_argument("--no-rerank", action="store_true")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_recommend)

    s = sub.add_parser("serve", help="start the HTTP API")
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 inside a container")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)
    try:
        return args.func(args, settings)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        print(f"invalid request: {first.get('msg', exc)}", file=sys.stderr)
        return 2
    except DownloadError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 4
    except (IndexValidationError, ArtifactError, ConfigError, OSError, RequestTimeout) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
