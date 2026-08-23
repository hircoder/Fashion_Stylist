#!/usr/bin/env bash
# ec2 bootstrap: amazon linux 2023 -> running service on :8000. logs to /var/log/stylist-boot.log
exec > /var/log/stylist-boot.log 2>&1
set -x
dnf install -y python3.12 git tar
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

mkdir -p /opt/stylist && cd /opt/stylist
aws s3 cp "s3://__BUCKET__/src/stylist-src.tar.gz" src.tar.gz
tar -xzf src.tar.gz && rm src.tar.gz
mkdir -p data
aws s3 cp "s3://__BUCKET__/index/stylist-index.tar.gz" index.tar.gz
tar -xzf index.tar.gz -C data && rm index.tar.gz

uv sync --frozen --no-dev --python 3.12
# pre-download the embedding model so /ready comes up without touching the hub later
HF_HOME=/opt/stylist/.hf ./.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a', device='cpu')"

cat > /etc/systemd/system/stylist.service <<'UNIT'
[Unit]
Description=fashion stylist api
After=network-online.target

[Service]
WorkingDirectory=/opt/stylist
Environment=HF_HOME=/opt/stylist/.hf
Environment=EMBED_DEVICE=cpu
Environment=LOG_LEVEL=INFO
Environment=LLM_PROVIDER=bedrock
Environment=BEDROCK_REGION=__BEDROCK_REGION__
Environment=LLM_MODEL=__LLM_MODEL__
Environment=PLANNER_BUDGET_S=0.35
Environment=RERANK_DEFAULT=0
Environment=SEMANTIC_PLAN_CACHE=1
Environment=TRUST_PROXY_HEADERS=1
Environment=RATE_LIMIT_PER_MINUTE=240
Environment=RESPONSE_CACHE_TTL_S=300
ExecStart=/opt/stylist/.venv/bin/uvicorn stylist.api:get_app --factory --host 0.0.0.0 --port 8000 --workers 2
ExecStartPost=-/bin/bash -c '/usr/local/bin/stylist-warmup.sh >> /var/log/stylist-warmup.log 2>&1 &'
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
cat > /usr/local/bin/stylist-warmup.sh <<'WARM'
#!/usr/bin/env bash
# two passes over the ui example queries. pass one starts the background planner
# calls, pass two (after they land) caches the planned answers. best effort only,
# a failure here must never take the unit down.
for _ in $(seq 1 120); do
  curl -sf -o /dev/null http://127.0.0.1:8000/ready && break
  sleep 1
done
run_pass() {
  while IFS= read -r q; do
    curl -sf -o /dev/null -X POST http://127.0.0.1:8000/recommend \
      -H 'content-type: application/json' \
      -d "{\"query\": \"$q\", \"k\": 3}" || true
  done <<'QS'
I need an outfit to go to the beach this summer
warm waterproof boots for hiking in the snow, under $80
what should my husband wear to an outdoor wedding in june
something cozy for working from home in winter
a gift for my 6 year old daughter who loves unicorns
smart casual outfit for a job interview at a startup
QS
}
run_pass
sleep 6
run_pass
echo "warmup done $(date)"
WARM
chmod +x /usr/local/bin/stylist-warmup.sh

systemctl daemon-reload
systemctl enable --now stylist
