# ============================================================================
#  2026 자율주행 AI챌린지 · 모션 플래닝 과제 (V-Max / Waymax)
#  RunPod Pod 용 커스텀 이미지
#
#  설계 원칙
#   1) 베이스는 nvidia/cuda:*-base — cuDNN/cuBLAS를 이미지에 넣지 않는다.
#      jax[cuda12]가 nvidia-*-cu12 pip 휠로 CUDA 12.9 + cuDNN 9.24를 직접
#      가져오므로, 이미지에 또 다른 cuDNN이 있으면 버전 충돌만 유발한다.
#   2) 무거운 의존성만 굽는다. V-Max 소스는 /workspace(네트워크 볼륨)에
#      클론해서 editable 설치 → 코드 수정할 때마다 재빌드할 필요 없음.
#   3) venv는 /opt/venv. RunPod은 /workspace를 볼륨으로 덮어쓰므로
#      /workspace 안에 런타임을 두면 안 된다.
#   4) 캐시(uv/HF/JAX 컴파일)는 /workspace로 보낸다. 컨테이너 디스크는
#      작고 파드를 지우면 사라지기 때문.
# ============================================================================

FROM nvidia/cuda:12.6.3-base-ubuntu24.04

LABEL org.opencontainers.image.title="vmax-runpod"
LABEL org.opencontainers.image.description="V-Max + Waymax + JAX(CUDA12) for the 2026 Autonomous Driving AI Challenge - Motion Planning"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# 1. OS 패키지
#    openssh-server를 미리 깔아두면 파드 부팅 후 바로 SSH가 열린다.
#    (RunPod 기본 안내처럼 start command에서 apt install 하면 접속까지 수 분 걸림)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        git-lfs \
        openssh-server \
        build-essential \
        pkg-config \
        rsync \
        unzip \
        zip \
        tmux \
        htop \
        vim \
        less \
        jq \
        tzdata \
        libgl1 \
        libglib2.0-0t64 \
        ffmpeg \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. uv + Python 3.11
#    가이드라인이 지정한 uv를 그대로 사용. Ubuntu 24.04 기본 파이썬은 3.12라
#    V-Max가 명시한 3.11을 uv managed python으로 따로 깐다.
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH

RUN uv python install 3.11 \
    && uv venv --python 3.11 /opt/venv

# ---------------------------------------------------------------------------
# 3. 파이썬 의존성 (락 파일로 고정 설치)
#    requirements.lock.txt는 uv pip compile 산출물이며 waymax git 커밋까지
#    핀되어 있다. 챌린지 '재현성 평가'를 대비해 락 파일을 그대로 커밋해 둘 것.
# ---------------------------------------------------------------------------
COPY requirements.in requirements.lock.txt /opt/build/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -r /opt/build/requirements.lock.txt

# ---------------------------------------------------------------------------
# 4. 런타임 환경변수
# ---------------------------------------------------------------------------
# JAX: 기본값은 GPU 메모리 75% 선점. TF 데이터로더/평가 스크립트와 충돌하므로 끈다.
#      (V-Max evaluate.py도 동일하게 false로 설정한다)
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform
# TensorFlow: waymax의 tfrecord 로딩 전용. CUDA 휠 없이 설치되어 CPU로 동작한다.
ENV TF_CPP_MIN_LOG_LEVEL=2 \
    TF_FORCE_GPU_ALLOW_GROWTH=true
# matplotlib: 헤드리스
ENV MPLBACKEND=Agg
# 캐시는 영구 볼륨으로 (컨테이너 디스크 절약 + 재부팅 후 재사용)
ENV XDG_CACHE_HOME=/workspace/.cache \
    UV_CACHE_DIR=/workspace/.cache/uv \
    HF_HOME=/workspace/.cache/huggingface \
    JAX_COMPILATION_CACHE_DIR=/workspace/.cache/jax
# NVIDIA container runtime (base 이미지가 이미 설정하지만 명시)
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---------------------------------------------------------------------------
# 5. SSH
# ---------------------------------------------------------------------------
RUN mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PermitUserEnvironment.*/PermitUserEnvironment yes/' /etc/ssh/sshd_config

# ---------------------------------------------------------------------------
# 6. 스크립트
# ---------------------------------------------------------------------------
COPY scripts/start.sh /usr/local/bin/start.sh
COPY scripts/bootstrap-workspace.sh /usr/local/bin/bootstrap-workspace.sh
COPY scripts/verify-env.py /usr/local/bin/verify-env.py

# Windows에서 빌드하면 스크립트에 CRLF 줄바꿈이 섞여 들어와
#   /usr/local/bin/start.sh: cannot execute: required file not found
# 로 컨테이너가 즉시 죽는다. 빌드 호스트가 무엇이든 안전하도록 LF로 정규화한다.
RUN sed -i 's/\r$//' /usr/local/bin/start.sh \
                     /usr/local/bin/bootstrap-workspace.sh \
                     /usr/local/bin/verify-env.py \
    && chmod +x /usr/local/bin/start.sh /usr/local/bin/bootstrap-workspace.sh

WORKDIR /workspace

EXPOSE 22

CMD ["/usr/local/bin/start.sh"]
