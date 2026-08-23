# common tasks. `make help` lists them.
PY      ?= uv run
INDEX_DIR ?= data/index
LIMIT   ?= 100000

.PHONY: help setup demo data ingest index index-full serve test test-all lint ui eval diagram index-tar clean

help:
	@grep -E '^[a-z-]+:.*## ' Makefile | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## create the venv and install everything (needs uv)
	uv sync --extra dev

demo: ## tiny 486 item index from the committed fixture, no download needed (~1 min)
	$(PY) stylist ingest --raw tests/fixtures/sample_500.jsonl.gz --out data/demo/catalog.parquet
	$(PY) stylist build-index --catalog data/demo/catalog.parquet --index-dir data/demo/index --sampling all
	@echo ""
	@echo "demo index ready. run:  INDEX_DIR=data/demo/index make serve"

data: ## download the raw Amazon Fashion metadata (224 MB)
	$(PY) stylist download-data

ingest: ## raw jsonl.gz -> data/processed/catalog.parquet (826K rows, ~30 s)
	$(PY) stylist ingest

index: ## embed + index the top $(LIMIT) items by rating count (~3-6 min)
	$(PY) stylist build-index --limit $(LIMIT) --sampling popular --index-dir $(INDEX_DIR)

index-full: ## embed + index the whole catalog (~20-45 min, ~3.3 GB RSS to serve)
	$(PY) stylist build-index --sampling all --index-dir data/index_full

serve: ## start the api + ui on :8000
	$(PY) stylist serve --port 8000

test: ## fast tests (no model download, no api keys)
	$(PY) pytest -m "not slow and not live" -q

test-all: ## everything, including the real embedding model and live llm (if keys set)
	$(PY) pytest -q

lint: ## ruff
	$(PY) ruff check src tests scripts && $(PY) ruff format --check src tests scripts

ui: ## rebuild the react bundle (needs node)
	cd ui && npm install && npm run build

eval: ## run the offline evaluation against $(INDEX_DIR) (the configs behind docs/evaluation.md)
	INDEX_DIR=$(INDEX_DIR) $(PY) python scripts/evaluate.py \
		--configs bm25,dense,hybrid,hybrid_nokw,hybrid_noquality,llm_plan,llm_plan_dense,llm_plan_bm25,llm_plan_rerank \
		--plan-cache docs/eval_plans.json

diagram: ## regenerate docs/architecture.svg from the script, then pdf + jpg (needs rsvg-convert, pillow)
	$(PY) python scripts/architecture_svg.py
	rsvg-convert -f pdf -o docs/architecture.pdf docs/architecture.svg
	rsvg-convert -f png -z 2 -o docs/architecture.png docs/architecture.svg
	$(PY) python -c "from PIL import Image; Image.open('docs/architecture.png').convert('RGB').save('docs/architecture.jpg', quality=92)"

index-tar: ## pack $(INDEX_DIR) for deployment and print its sha256
	tar -czf data/index.tar.gz -C $(dir $(INDEX_DIR)) $(notdir $(INDEX_DIR))
	shasum -a 256 data/index.tar.gz

clean:
	rm -rf .pytest_cache .ruff_cache
