# cpu-only image for railway / any container host.
#
# By default the build bakes a real demo index into the image: it downloads the raw
# Amazon Fashion metadata (224 MB), ingests it and embeds the BAKE_INDEX_LIMIT most rated
# listings (40K by default, ~4 min on a cpu builder). The running container then needs no
# volume and no INDEX_URL. Build with --build-arg BAKE_INDEX_LIMIT=0 to skip that (ci does)
# and provide an index through a volume at /app/data or INDEX_URL + INDEX_SHA256 instead.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/app/.hf \
    EMBED_DEVICE=cpu \
    LOG_LEVEL=INFO

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

# deps first (cached layer), project after
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY ui/dist ./ui/dist
COPY docs/overview.html ./docs/overview.html
COPY tests/fixtures ./tests/fixtures
RUN uv sync --frozen --no-dev

# pin + pre-download the embedding model so boot does not depend on the hub
ARG EMBEDDING_REVISION=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
ENV EMBEDDING_REVISION=${EMBEDDING_REVISION}
RUN /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5', revision='${EMBEDDING_REVISION}', device='cpu')"

# optional: bake a demo index (see the note at the top). STRICT_DATA_SHA=0 lets the
# build survive an upstream refresh of the raw file (the checksum becomes advisory).
ARG BAKE_INDEX_LIMIT=40000
ARG STRICT_DATA_SHA=1
RUN if [ "${BAKE_INDEX_LIMIT}" -gt 0 ]; then \
      /app/.venv/bin/stylist download-data $([ "${STRICT_DATA_SHA}" = "1" ] && echo --strict) \
      && /app/.venv/bin/stylist ingest \
      && /app/.venv/bin/stylist build-index --limit "${BAKE_INDEX_LIMIT}" --sampling popular \
      && rm -rf /app/data/raw /app/data/processed; \
    else mkdir -p /app/data; fi

# run as a normal user: the code and the virtualenv stay owned by root (read-only for
# the service), only the data dir and the model cache are writable by it
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/.hf \
    && chown -R app:app /app/data /app/.hf
USER app
# container hosts normally put a proxy in front of the service, so the client ip arrives
# in x-forwarded-for. Exposing the container directly? build with TRUST_PROXY=0 (or just
# set TRUST_PROXY_HEADERS=0 on the service) or clients can spoof their way past the limiter.
ARG TRUST_PROXY=1
ENV TRUST_PROXY_HEADERS=${TRUST_PROXY}

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready').status==200 else 1)"
CMD ["/app/.venv/bin/stylist", "serve", "--host", "0.0.0.0"]
