FROM python:3.11-slim

WORKDIR /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG INSTALL_EXTRAS=

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TORCH_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch torchvision && \
    if [ -n "${INSTALL_EXTRAS}" ]; then \
      apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* && \
      pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"; \
    else \
      pip install --no-cache-dir .; \
    fi

COPY app ./app

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --limit-concurrency 32"]
