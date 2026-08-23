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
Environment=BEDROCK_REGION=us-east-1
Environment=PLANNER_BUDGET_S=0.35
Environment=RERANK_DEFAULT=0
Environment=SEMANTIC_PLAN_CACHE=1
Environment=TRUST_PROXY_HEADERS=1
Environment=RATE_LIMIT_PER_MINUTE=240
ExecStart=/opt/stylist/.venv/bin/uvicorn stylist.api:get_app --factory --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now stylist
