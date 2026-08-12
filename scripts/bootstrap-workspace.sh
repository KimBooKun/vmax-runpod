#!/usr/bin/env bash
# ============================================================================
# /workspace(영구 볼륨) 초기화
#  - V-Max 소스 클론 + editable 설치 (--no-deps: 의존성은 이미지에 이미 있음)
#  - 데이터/체크포인트 디렉터리 생성
# 멱등(idempotent)하게 동작하므로 재부팅마다 실행돼도 안전하다.
# ============================================================================
set -euo pipefail

VMAX_DIR="${VMAX_DIR:-/workspace/V-Max}"
VMAX_REPO="${VMAX_REPO:-https://github.com/valeoai/V-Max.git}"
VMAX_REF="${VMAX_REF:-main}"
PY=/opt/venv/bin/python

echo "[bootstrap] target=$VMAX_DIR"

mkdir -p /workspace/data/{train,valid,test} /workspace/runs /workspace/checkpoints

if [[ ! -d "$VMAX_DIR/.git" ]]; then
  echo "[bootstrap] cloning V-Max ($VMAX_REF)"
  git clone --branch "$VMAX_REF" "$VMAX_REPO" "$VMAX_DIR"
else
  echo "[bootstrap] V-Max already present, skipping clone"
fi

# editable 설치 확인 (site-packages에 링크가 없으면 설치)
if ! "$PY" -c "import vmax" >/dev/null 2>&1; then
  echo "[bootstrap] installing V-Max in editable mode"
  uv pip install --python "$PY" --no-deps -e "$VMAX_DIR"
else
  echo "[bootstrap] vmax importable, skipping install"
fi

# 챌린지 데이터 위치 안내용 심볼릭 링크
# V-Max 기본 경로 규약: data/train/*.tfrecord
if [[ ! -e "$VMAX_DIR/data" ]]; then
  ln -s /workspace/data "$VMAX_DIR/data"
  echo "[bootstrap] linked $VMAX_DIR/data -> /workspace/data"
fi

echo "[bootstrap] done"
echo "[bootstrap] 검증: python /usr/local/bin/verify-env.py"
