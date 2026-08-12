#!/usr/bin/env python
"""환경 검증 스크립트.

파드에 들어가서 제일 먼저 이걸 돌린다. 확인 항목:
  1. JAX가 GPU를 잡는가 (jaxlib CUDA 플러그인 정상 로드)
  2. TensorFlow가 GPU를 잡지 '않는가' (tfrecord 로딩 전용, GPU 메모리 뺏으면 안 됨)
  3. waymax / vmax import
  4. 간단한 GPU matmul + JIT 컴파일 왕복 시간
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

OK, FAIL, WARN = "\033[32m OK \033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"
failed = False


def line(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


# --------------------------------------------------------------------------
# 1. JAX / GPU
# --------------------------------------------------------------------------
try:
    import jax
    import jaxlib

    devices = jax.devices()
    line(OK, f"jax {jax.__version__} / jaxlib {jaxlib.__version__}")
    line(OK, f"jax.devices() = {devices}")
    if not any(d.platform == "gpu" for d in devices):
        line(FAIL, "GPU가 안 보입니다. 파드에 GPU가 붙었는지, 드라이버 버전(>=525)을 확인하세요.")
        failed = True
    else:
        d = next(d for d in devices if d.platform == "gpu")
        line(OK, f"GPU: {d.device_kind}")
except Exception as e:  # noqa: BLE001
    line(FAIL, f"jax import 실패: {e}")
    failed = True

# --------------------------------------------------------------------------
# 2. TensorFlow는 CPU 전용이어야 한다
# --------------------------------------------------------------------------
try:
    import tensorflow as tf

    tf_gpus = tf.config.list_physical_devices("GPU")
    line(OK, f"tensorflow {tf.__version__}")
    if tf_gpus:
        line(WARN, f"TF가 GPU를 잡았습니다({tf_gpus}). JAX와 메모리 경합 가능 → "
                   "학습 스크립트 최상단에 tf.config.set_visible_devices([], 'GPU') 추가 권장")
    else:
        line(OK, "TF는 CPU 전용 (의도한 동작 - tfrecord 로딩에만 사용)")
except Exception as e:  # noqa: BLE001
    line(FAIL, f"tensorflow import 실패: {e}")
    failed = True

# --------------------------------------------------------------------------
# 3. waymax / vmax
# --------------------------------------------------------------------------
for mod in ("waymax", "vmax"):
    try:
        m = __import__(mod)
        line(OK, f"{mod} import 성공 ({getattr(m, '__file__', '?')})")
    except Exception as e:  # noqa: BLE001
        tag = FAIL if mod == "waymax" else WARN
        line(tag, f"{mod} import 실패: {e}"
                  + ("  → bootstrap-workspace.sh 를 실행하세요" if mod == "vmax" else ""))
        if mod == "waymax":
            failed = True

# --------------------------------------------------------------------------
# 4. GPU 연산 + JIT 왕복
# --------------------------------------------------------------------------
if not failed:
    try:
        import jax.numpy as jnp

        key = jax.random.PRNGKey(0)
        a = jax.random.normal(key, (4096, 4096), dtype=jnp.float32)

        f = jax.jit(lambda x: (x @ x.T).sum())
        t0 = time.perf_counter()
        f(a).block_until_ready()
        t_compile = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(10):
            f(a).block_until_ready()
        t_run = (time.perf_counter() - t0) / 10

        line(OK, f"JIT 컴파일 {t_compile * 1e3:.0f} ms / 실행 {t_run * 1e3:.2f} ms "
                 f"(4096x4096 matmul ≈ {2 * 4096**3 / t_run / 1e12:.1f} TFLOPS)")
    except Exception as e:  # noqa: BLE001
        line(FAIL, f"GPU 연산 실패: {e}")
        failed = True

print()
if failed:
    print("환경 검증 실패 — 위 FAIL 항목을 먼저 해결하세요.")
    sys.exit(1)
print("환경 검증 통과. 학습을 시작해도 됩니다.")
