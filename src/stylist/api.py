"""HTTP surface: FastAPI app with /recommend, /health, /ready and the bundled web UI.

The index is loaded once at startup. If loading fails the process still starts (so
/health works and the error is visible) but /ready returns 503 and /recommend refuses.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from stylist import __version__
from stylist.artifacts import ArtifactError
from stylist.config import ConfigError, Settings, configure_logging
from stylist.embeddings import make_embedder
from stylist.index import IndexValidationError, SearchIndex
from stylist.llm import make_llm_client
from stylist.schemas import ErrorBody, RecommendRequest, RecommendResponse
from stylist.service import RecommendationService, RequestTimeout

log = logging.getLogger(__name__)

UI_DIST = Path(__file__).resolve().parents[2] / "ui" / "dist"
OVERVIEW_PAGE = Path(__file__).resolve().parents[2] / "docs" / "overview.html"

DESCRIPTION = """Semantic product recommendations for a fashion catalog.

Send a natural-language request ("I need an outfit to go to the beach this summer")
and get back product picks grouped by outfit slot. Under the hood: an LLM turns the
request into a retrieval plan, hybrid search (dense embeddings + BM25) finds candidates
per slot inside the price/audience constraints, and the LLM reranks and explains.
"""


_RX_PATH = re.compile(r"(/[\w.\-]+){2,}")
_RX_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|(?i:api[_-]?key|token|secret|password)\s*[=:]\s*\S+)"
)


def _scrub(text: str) -> str:
    """Strip local filesystem paths and anything that looks like a credential out of text
    that goes to a client."""
    text = _RX_SECRET.sub("<redacted>", text)
    return _RX_PATH.sub("<path>", text)[:300]


def _public_error(exc: Exception) -> str:
    """Error text safe to show to any client: class name + message without local paths."""
    return f"{type(exc).__name__}: {_scrub(str(exc))}"


def _startup_error(exc: Exception) -> str:
    """A curated sentence for /health and /ready. The full exception goes to the log only:
    startup failures quote paths, urls and sometimes configuration values."""
    if isinstance(exc, IndexValidationError):
        return f"index problem: {_scrub(str(exc))} (build it with make index, or set INDEX_URL)"
    if isinstance(exc, ArtifactError):
        return "index download or install failed; check INDEX_URL, INDEX_SHA256 and the logs"
    if isinstance(exc, ConfigError):
        return "configuration error, see the server logs"
    return f"startup failed ({type(exc).__name__}), see the server logs"


def _error(status: int, code: str, message: str) -> JSONResponse:
    body = ErrorBody(error={"code": code, "message": message})  # type: ignore[arg-type]
    return JSONResponse(status_code=status, content=body.model_dump())


class IndexNotLoaded(Exception):
    pass


def build_service(settings: Settings) -> RecommendationService:
    """Load everything the service needs. Raises on any problem (caller decides)."""
    from stylist.artifacts import ensure_index

    ensure_index(settings)
    index = SearchIndex.load(settings.index_dir, expected_model=settings.embedding_name)
    embedder = make_embedder(settings)
    if embedder.dim != index.meta.dim:
        raise IndexValidationError(
            f"embedder dim {embedder.dim} != index dim {index.meta.dim}, rebuild the index"
        )
    revision = getattr(embedder, "revision", None)
    if revision != index.meta.embedding_revision:
        # the weights that encode queries must be the weights that encoded the documents;
        # a missing value on either side is a mismatch, not a pass
        raise IndexValidationError(
            f"embedding revision {revision!r} != index revision "
            f"{index.meta.embedding_revision!r}, rebuild the index with this model"
        )
    llm = make_llm_client(settings)
    rerank_llm = (
        make_llm_client(settings, settings.llm_rerank_model) if settings.llm_rerank_model else None
    )
    log.info(
        "service ready: %d rows (%s), embedder=%s, llm=%s/%s",
        index.n_rows,
        index.meta.sampling,
        embedder.name,
        llm.provider if llm else "none",
        llm.model if llm else "-",
    )
    return RecommendationService(index, embedder, settings, llm, rerank_llm=rerank_llm)


class _TokenBucket:
    """Per-client token bucket: `rate` requests per minute with the same burst size.
    Buckets are kept for the most recent clients only (bounded memory)."""

    def __init__(self, per_minute: int, max_clients: int = 10_000):
        self.rate = per_minute / 60.0
        # a sixth of the minute at once (10 at 60/min): enough for a page load, not a flood
        self.burst = float(max(1, per_minute // 6))
        self.max_clients = max_clients
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """(allowed, seconds until the next token)."""
        now = time.monotonic() if now is None else now
        tokens, last = self._buckets.pop(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        allowed = tokens >= 1.0
        if allowed:
            tokens -= 1.0
        self._buckets[key] = (tokens, now)
        while len(self._buckets) > self.max_clients:
            self._buckets.popitem(last=False)
        return allowed, (0.0 if allowed else (1.0 - tokens) / self.rate)


def _client_ip(request: Request, trust_proxy: bool) -> str:
    """The rate-limit key. x-forwarded-for is client-controlled unless a proxy in front of
    us overwrites it, so it is only honoured behind one (TRUST_PROXY_HEADERS=1)."""
    if trust_proxy:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()[:64]
    return request.client.host if request.client else "unknown"


# image hosts mirror schemas.safe_image_url: what the api hands out, the page may load
_CSP = (
    "default-src 'self'; img-src 'self' data: https://*.media-amazon.com "
    "https://*.ssl-images-amazon.com https://*.images-amazon.com; "
    "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
    "frame-ancestors 'none'"
)


class _SecurityHeaders:
    """Pure ASGI middleware adding the usual browser protections to every response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        html = path in ("/", "/overview")

        async def send_wrapped(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers += [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                ]
                if html:
                    headers.append((b"content-security-policy", _CSP.encode()))
                message = {**message, "headers": headers}
            await send(message)

        return await self.app(scope, receive, send_wrapped)


class _BodyLimit:
    """Pure ASGI middleware: 413 for request bodies above `max_bytes`. Content-Length is
    checked first; chunked bodies are buffered up to the cap (request bodies here are a
    few hundred bytes of json)."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        length = headers.get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.max_bytes:
            return await self._reject(send)
        body = b""
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return  # the client left mid-body: nothing to answer
            body += message.get("body", b"")
            more = message.get("more_body", False)
            if len(body) > self.max_bytes:
                return await self._reject(send)
        replayed = False

        async def replay():
            # the whole body once, then an empty terminator forever (never the drained socket)
            nonlocal replayed
            if replayed:
                return {"type": "http.request", "body": b"", "more_body": False}
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        return await self.app(scope, replay, send)

    async def _reject(self, send):
        body = (
            ErrorBody(
                error={
                    "code": "payload_too_large",
                    "message": f"request body above {self.max_bytes} bytes",
                }  # type: ignore[arg-type]
            )
            .model_dump_json()
            .encode()
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def create_app(settings: Settings | None = None, *, service: RecommendationService | None = None):
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        app.state.load_error = None
        if app.state.service is None:
            try:
                app.state.service = build_service(settings)
            except Exception as exc:  # noqa: BLE001 - anything here must keep /health alive
                log.exception("service failed to start")
                app.state.load_error = _startup_error(exc)
                if settings.startup_fail_fast:
                    raise
        yield
        if app.state.service is not None:
            app.state.service.close()

    app = FastAPI(
        title="Fashion stylist recommendation API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["content-type"],
        )
    app.add_middleware(_BodyLimit, max_bytes=settings.max_body_bytes)
    app.add_middleware(_SecurityHeaders)
    bucket = (
        _TokenBucket(settings.rate_limit_per_minute) if settings.rate_limit_per_minute else None
    )
    inflight = (
        asyncio.Semaphore(settings.max_inflight_requests)
        if settings.max_inflight_requests
        else None
    )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        msg = first.get("msg", "invalid request")
        msg = msg.removeprefix("Value error, ")
        return _error(422, "validation_error", f"{loc}: {msg}" if loc else msg)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, f"http_{exc.status_code}", _scrub(str(exc.detail)))

    @app.exception_handler(IndexNotLoaded)
    async def _not_loaded(_: Request, exc: IndexNotLoaded):
        return _error(503, "index_not_loaded", str(exc))

    @app.exception_handler(RequestTimeout)
    async def _timeout(_: Request, exc: RequestTimeout):
        return _error(504, "deadline_exceeded", str(exc))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return _error(500, "internal_error", "something went wrong, see server logs")

    def _service(request: Request) -> RecommendationService:
        svc = request.app.state.service
        if svc is None:
            raise IndexNotLoaded(request.app.state.load_error or "index not loaded")
        return svc

    @app.post(
        "/recommend",
        response_model=RecommendResponse,
        tags=["recommend"],
        summary="Recommend products for a natural-language request",
        responses={
            413: {"model": ErrorBody},
            422: {"model": ErrorBody},
            429: {"model": ErrorBody},
            503: {"model": ErrorBody},
            504: {"model": ErrorBody},
        },
    )
    async def recommend(body: RecommendRequest, request: Request):
        if bucket is not None:
            allowed, wait = bucket.allow(_client_ip(request, settings.trust_proxy_headers))
            if not allowed:
                resp = _error(429, "rate_limited", "too many requests from this client, slow down")
                resp.headers["Retry-After"] = str(max(1, int(wait + 0.999)))
                return resp
        # asyncio runs one task at a time and Semaphore.acquire() does not yield when a
        # permit is free, so this check-then-acquire pair cannot be interleaved by another
        # request: the cap holds. (It would not under threads.)
        if inflight is not None and inflight.locked():
            resp = _error(503, "busy", "the service is at its concurrency limit, retry shortly")
            resp.headers["Retry-After"] = "2"
            return resp
        if inflight is None:
            return await _service(request).recommend(body)
        async with inflight:
            return await _service(request).recommend(body)

    @app.get("/health", tags=["ops"], summary="Liveness + what is loaded (never fails)")
    async def health(request: Request) -> dict:
        svc = request.app.state.service
        index = None
        if svc is not None:
            m = svc.index.meta
            index = {"rows": svc.index.n_rows, "sampling": m.sampling, "model": m.embedding_model}
        return {
            "status": "ok",
            "version": __version__,
            "index_loaded": svc is not None,
            "index": index,
            "load_error": request.app.state.load_error,
            "llm": {
                "provider": svc.llm.provider if svc and svc.llm else None,
                "model": svc.llm.model if svc and svc.llm else None,
            },
        }

    @app.get("/ready", tags=["ops"], summary="Readiness: 200 only when the index is loaded")
    async def ready(request: Request):
        if request.app.state.service is None:
            return _error(503, "index_not_loaded", request.app.state.load_error or "loading")
        return {"ready": True}

    if (UI_DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")

    @app.get("/overview", include_in_schema=False)
    async def overview():
        """The walkthrough deck (docs/overview.html): scope, data, architecture, samples."""
        if OVERVIEW_PAGE.exists():
            return FileResponse(OVERVIEW_PAGE)
        return HTMLResponse("<p>overview page not found (docs/overview.html)</p>", status_code=404)

    @app.get("/", include_in_schema=False)
    async def root():
        index_html = UI_DIST / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        return HTMLResponse(
            "<h1>Fashion stylist API</h1><p>UI not built. Try <a href='/docs'>/docs</a>.</p>"
        )

    return app


def get_app():
    """uvicorn entry point: `uvicorn stylist.api:get_app --factory`."""
    return create_app()
