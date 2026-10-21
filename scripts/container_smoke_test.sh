#!/usr/bin/env bash
# Build an image target, start it, and check it end to end:
#   scripts/container_smoke_test.sh [lite|full]
# Exits non-zero on any failure; the container is always removed.
set -euo pipefail

TARGET="${1:-lite}"
IMAGE="fcagents:${TARGET}-smoke"
PORT="${SMOKE_PORT:-7870}"
cd "$(dirname "$0")/.."

echo "==> building target '${TARGET}'"
docker build --target "${TARGET}" -t "${IMAGE}" .

echo "==> starting container on port ${PORT}"
cid=$(docker run -d -p "${PORT}:7860" --health-interval=3s --health-start-period=5s "${IMAGE}")
trap 'docker rm -f "${cid}" >/dev/null' EXIT

echo "==> waiting for Docker health status"
for _ in $(seq 1 60); do
  status=$(docker inspect -f '{{.State.Health.Status}}' "${cid}")
  [ "${status}" = "healthy" ] && break
  if [ "$(docker inspect -f '{{.State.Running}}' "${cid}")" != "true" ]; then
    docker logs "${cid}"; echo "container exited"; exit 1
  fi
  sleep 2
done
[ "${status}" = "healthy" ] || { docker logs "${cid}"; echo "not healthy: ${status}"; exit 1; }
echo "    healthy"

echo "==> app page reachable from the host"
curl -sf -o /dev/null "http://localhost:${PORT}/"
echo "    HTTP 200"

echo "==> running every sample through the agents inside the container"
docker exec -i "${cid}" python - <<'PY'
from pathlib import Path
import pandas as pd
from fcagents.orchestrator import run_pipeline

expected = {  # sample -> {agent: decision}
    "hourly_energy.csv": {"frequency": "h"},
    "15min_with_duplicates.csv": {"frequency": "15min"},
    "daily_sales_with_gaps.csv": {"regularity": "regular_with_gaps", "frequency": "D"},
    "business_day_stock.csv": {"frequency": "B"},
    "monthly_passengers.csv": {"frequency": "MS"},
    "multi_store_long.csv": {"frequency": "D"},
    "irregular_sensor.csv": {"regularity": "irregular", "resampling": "resampled"},
}
failures = 0
for name, exp in expected.items():
    got = {r.agent: r.decision for r, _ in run_pipeline(pd.read_csv(Path("data/samples") / name))}
    ok = all(got.get(k) == v for k, v in exp.items())
    failures += not ok
    print(f"    {'ok ' if ok else 'BAD'} {name}: {got}")
weekly = {r.agent: r.decision for r, _ in run_pipeline(pd.read_csv("data/samples/irregular_sensor.csv"), resample_freq="W")}
ok = weekly["resampling"] == "too_few_points" and weekly["frequency"] == "skipped"
failures += not ok
print(f"    {'ok ' if ok else 'BAD'} irregular_sensor.csv @W -> too_few_points warning")
raise SystemExit(failures)
PY

if [ "${TARGET}" = "full" ]; then
  echo "==> baked-in LLM loads and generates offline"
  docker exec -e HF_HUB_OFFLINE=1 "${cid}" python -c "
from fcagents.llm.backend import TransformersBackend
out = TransformersBackend().generate('Answer in one word.', 'Reply with the word OK.', max_new_tokens=5)
print('    model said:', repr(out)); assert out
"
fi

echo "==> smoke test passed (${TARGET})"
