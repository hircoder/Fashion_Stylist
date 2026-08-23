"""HTTP surface: FastAPI app with /recommend, /health, /ready and the bundled web UI.

The index is loaded once at startup. If loading fails the process still starts (so
/health works and the error is visible) but /ready returns 503 and /recommend refuses.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from stylist import __version__
from stylist.config import Settings
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


def _scrub(text: str) -> str:
    """Strip local filesystem paths out of text that goes to a client."""
    import re

    return re.sub(r"(/[\w.\-]+){2,}", "<path>", text)[:300]


def _public_error(exc: Exception) -> str:
    """Error text safe to show to any client: class name + message without local paths."""
    return f"{type(exc).__name__}: {_scrub(str(exc))}"


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
    if revision and index.meta.embedding_revision and revision != index.meta.embedding_revision:
        raise IndexValidationError(
            f"embedding revision {revision} != index revision {index.meta.embedding_revision}"
        )
    llm = make_llm_client(settings)
    log.info(
        "service ready: %d rows (%s), embedder=%s, llm=%s/%s",
        index.n_rows,
        index.meta.sampling,
        embedder.name,
        llm.provider if llm else "none",
        llm.model if llm else "-",
    )
    return RecommendationService(index, embedder, settings, llm)


def create_app(settings: Settings | None = None, *, service: RecommendationService | None = None):
    settings = settings or Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        app.state.load_error = None
        if app.state.service is None:
            try:
                app.state.service = build_service(settings)
            except Exception as exc:  # noqa: BLE001 - anything here must keep /health alive
                log.exception("service failed to start")
                app.state.load_error = _public_error(exc)
        yield

    app = FastAPI(
        title="Fashion stylist recommendation API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
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
            422: {"model": ErrorBody},
            503: {"model": ErrorBody},
            504: {"model": ErrorBody},
        },
    )
    async def recommend(body: RecommendRequest, request: Request) -> RecommendResponse:
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
