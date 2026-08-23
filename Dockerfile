# cpu-only image for railway / any container host. the index is NOT baked in:
# either mount a volume with data/index or set INDEX_URL + INDEX_SHA256 (see README).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/app/.hf \
    EMBED_DEVICE=cpu

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

# deps first (cached layer), project after
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY ui/dist ./ui/dist
COPY tests/fixtures ./tests/fixtures
RUN uv sync --frozen --no-dev

# pin + pre-download the embedding model so boot does not depend on the hub
ARG EMBEDDING_REVISION=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
ENV EMBEDDING_REVISION=${EMBEDDING_REVISION}
RUN uv run python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5', revision='${EMBEDDING_REVISION}', device='cpu')"

# run as a normal user; only the data dir and the model cache need to be writable
RUN useradd --create-home --uid 10001 app && mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready').status==200 else 1)"
CMD ["uv", "run", "stylist", "serve", "--host", "0.0.0.0", "--port", "8000"]
