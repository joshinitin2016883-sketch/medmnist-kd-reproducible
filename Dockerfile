# syntax=docker/dockerfile:1.7
#
# Containerises the Flask API in app.py.
#
# Build:  docker build -t medmnist-kd-api .
# Run:    docker run --rm -p 8000:8000 \
#             -v "${PWD}/checkpoints:/models:ro" \
#             -e MODEL_PATH=/models/baseline_resnet_derma_seed42.pt \
#             medmnist-kd-api
#
# ---------------------------------------------------------------------------
# BASE IMAGE PINNING
#
# Pinned by digest, not just by tag. `python:3.12-slim-bookworm` is a moving
# target -- the same tag resolves to different bytes as the upstream image is
# rebuilt for security patches, so a build that worked in March can fail in
# June with no change to this file. A digest is immutable: this build produces
# the same base image forever.
#
# The trade-off is that you no longer receive those security patches
# automatically. The digest must be bumped deliberately, which is the point:
# base image updates become a reviewed change rather than a silent one.
#
# Python 3.12 rather than the 3.14 used for development: 3.12 has the widest
# wheel availability, and serving needs no 3.14-specific behaviour. Wheels are
# pinned by version in requirements-serve.txt, so the two environments install
# the same package versions.
# ---------------------------------------------------------------------------
ARG BASE=python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

# ===========================================================================
# Stage 1 -- builder
#
# Dependencies are installed into a self-contained virtualenv here, and only
# that virtualenv is carried into the final image. pip's HTTP cache, its
# wheel-build scratch space, and any transient build tooling stay behind in a
# layer that is never shipped. For a torch install that is a few hundred MB of
# difference in the published image.
# ===========================================================================
FROM ${BASE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# -------------------------------------------------------------------------
# WHY requirements-serve.txt IS COPIED BEFORE THE SOURCE CODE
#
# Docker caches each instruction as a layer, keyed on the instruction and the
# contents of whatever it touches. A layer is reused only if it and every
# layer before it are unchanged.
#
# The pip install below is by far the most expensive step -- ~200 MB of torch
# over the network, tens of seconds at best.
#
# If we wrote `COPY . .` first, the build context (every source file) would be
# baked into a layer *above* the cache and *below* the install. Editing one
# character in app.py would invalidate that layer, and every layer after it,
# forcing a full reinstall of torch on every single build.
#
# By copying only the dependency manifest first, the install layer depends on
# nothing but that manifest. Source edits invalidate only the cheap COPY layer
# near the end. torch is reinstalled when, and only when, the dependency list
# actually changes.
#
# The general rule: order instructions from least to most frequently changing.
# -------------------------------------------------------------------------
COPY requirements-serve.txt .

# CPU-only torch. The CUDA build is ~2.6 GB against ~200 MB here, and a
# container serving one image per request has no use for a GPU. PyPI would
# quietly serve a different build, so the index is specified explicitly.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements-serve.txt \
 && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
 && find /opt/venv -name "*.pyc" -delete

# ===========================================================================
# Stage 2 -- runtime
# ===========================================================================
FROM ${BASE} AS runtime

LABEL org.opencontainers.image.title="medmnist-kd-api" \
      org.opencontainers.image.description="Flask API serving a MedMNIST classifier with Grad-CAM" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/joshinitin2016883-sketch/medmnist-kd-reproducible"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/models/model.pt \
    DEVICE=cpu \
    PORT=8000

# A container process should not run as root. If the application is
# compromised, an attacker inherits an unprivileged account rather than root
# inside the container -- which, combined with a kernel escape, is the
# difference between a contained incident and a host compromise.
RUN useradd --create-home --uid 10001 appuser

# The virtualenv arrives as a single directory. It is relocatable because
# nothing outside /opt/venv is referenced, and PATH above puts its bin first.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Source last: the layer that changes on nearly every build sits above every
# expensive layer, so a code edit rebuilds only this.
COPY --chown=appuser:appuser app.py evaluate.py ./

USER appuser
EXPOSE 8000

# ---------------------------------------------------------------------------
# THE MODEL IS NOT BAKED INTO THE IMAGE
#
# Checkpoints are ~90 MB and change far more often than the serving code. Two
# reasons to mount rather than COPY:
#
#   1. Size and churn. Baking one in would add 90 MB to every image and mean a
#      new image build for every retrain, even with identical code.
#   2. Coupling. Image version and model version become independent, so the
#      same image can serve any checkpoint -- which is what makes A/B and
#      rollback of a model possible without a rebuild.
#
# The cost is that the image is not self-contained: it will not start without a
# mounted model. /health returns 503 in that case rather than serving nonsense.
# ---------------------------------------------------------------------------
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# gunicorn, not `flask run`. The Flask development server is single-threaded,
# has no request timeouts, and is explicitly documented as unsuitable for
# production.
#
# --preload imports the app once before forking. app.py loads the checkpoint at
# import time, so the model is read from disk once and the workers share those
# pages copy-on-write, instead of each worker loading its own ~100 MB copy.
#
# Worker count here is bound by memory, not CPU. Raise it only alongside the
# container memory limit.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", \
     "--workers", "2", "--threads", "2", \
     "--preload", "--timeout", "60", \
     "--access-logfile", "-", "--error-logfile", "-", \
     "app:app"]
