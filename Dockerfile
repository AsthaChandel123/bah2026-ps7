# exopipe -- one image, two Cloud Run shapes (dashboard service + batch job).
#
#   docker build -t exopipe .
#   docker run -p 8080:8080 exopipe            # dashboard (default)
#   docker run exopipe run --input <csv|dir>   # sector-scale batch job
#
# Base: slim CPython. batman-package / emcee compile small C extensions on
# install, so a C toolchain is needed at build time only.
FROM python:3.11-slim AS runtime

# --- System build deps (compiler for batman/emcee), then purge apt lists --- #
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# --- Dependency layer (cache-friendly) ------------------------------------ #
# Copy only the packaging metadata + source needed to resolve/install deps so
# that edits to app/, examples/, configs/ etc. don't bust the heavy pip layer.
COPY pyproject.toml README.md ./
COPY src ./src

# science + ml + app extras: detrend/search/fit/classify, the Streamlit
# dashboard, and lightkurve + astroquery so the batch job can fetch real TESS
# data. Larger image, but it covers both runtime shapes from one build.
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -e ".[science,ml,app]"

# --- Application assets ---------------------------------------------------- #
# Copied after the dependency layer so code/asset changes are cheap rebuilds.
COPY app ./app
COPY configs ./configs
COPY examples ./examples
COPY models ./models
COPY report/template.md ./report/template.md
COPY docker ./docker

# --- Non-root runtime user ------------------------------------------------- #
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

ENTRYPOINT ["bash", "docker/entrypoint.sh"]
# Default shape: the Streamlit dashboard service. Cloud Run Jobs override this
# with `--args run,--input,...` (or set EXOPIPE_MODE=job).
CMD ["dashboard"]
