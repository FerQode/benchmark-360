# ─────────────────────────────────────────────────────────────
# Benchmark 360 — Container de Producción
# Multi-stage build: Instala dependencias en una etapa,
# copia solo lo necesario al contenedor final.
# ─────────────────────────────────────────────────────────────

# Etapa 1: Builder — instala dependencias pesadas
FROM python:3.12-slim AS builder

WORKDIR /app

# Instalar dependencias del sistema para Playwright y compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv (gestor de paquetes requerido por el reto)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copiar solo el manifiesto primero (cache de Docker layers)
COPY pyproject.toml ./

# Instalar dependencias Python
RUN uv pip install --system --no-cache -r pyproject.toml

# Instalar Playwright + Chromium (para scraping de SPAs)
RUN playwright install chromium --with-deps


# Etapa 2: Runtime — imagen final liviana
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copiar dependencias instaladas desde builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /root/.cache /root/.cache
COPY --from=builder /usr/lib /usr/lib

# Reinstalar playwright browsers en runtime
RUN playwright install chromium --with-deps 2>/dev/null || true

# Copiar codigo fuente del proyecto
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/input/ ./data/input/
COPY run_pipeline.py ./
COPY pyproject.toml ./

# Crear directorios de salida
RUN mkdir -p data/output/charts data/cache data/logs

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CACHE_TTL_HOURS=24 \
    MAX_CONCURRENT_SCRAPERS=3 \
    REQUEST_DELAY_MIN=2.0 \
    REQUEST_DELAY_MAX=5.0

# Las API keys se pasan en runtime via --env-file o -e
# NUNCA hardcodear secrets en la imagen

# Healthcheck: verificar que Python y las dependencias cargan
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from src.config.settings import get_settings; print('OK')" || exit 1

# Comando por defecto: ejecutar pipeline completo
# Flujo: Pipeline → ML Enrichment → Notebook
CMD ["sh", "-c", "\
    echo '=== Benchmark 360 — Ejecucion Containerizada ===' && \
    python run_pipeline.py || python scripts/build_demo_dataset.py && \
    python scripts/enrich_and_export.py && \
    python scripts/generate_notebook.py && \
    echo '=== Pipeline completado ===' \
"]
