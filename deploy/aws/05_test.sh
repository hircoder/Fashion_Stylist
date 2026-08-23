#!/usr/bin/env bash
# latency runs against the distribution (or any base url). writes one json per config
# into experiments/. usage: ./05_test.sh https://dxxxx.cloudfront.net label
set -euo pipefail
cd "$(dirname "$0")"
BASE=${1:?base url}
LABEL=${2:?label}
N=${N:-30}
python3 ./latency_probe.py --base "$BASE" --label "$LABEL" --n "$N" --out "experiments/${LABEL}.json"
