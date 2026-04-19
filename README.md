# Benchmark 360

**Inteligencia Competitiva Automatizada para el Mercado de Internet Fijo en Ecuador**

[![CI Tests](https://github.com/FerQode/benchmark-360/actions/workflows/tests.yml/badge.svg)](https://github.com/FerQode/benchmark-360/actions/workflows/tests.yml)
[![Daily Pipeline](https://github.com/FerQode/benchmark-360/actions/workflows/daily_benchmark.yml/badge.svg)](https://github.com/FerQode/benchmark-360/actions/workflows/daily_benchmark.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Que es Benchmark 360?

Un pipeline automatizado que extrae, procesa y analiza diariamente la oferta
comercial de los principales ISPs de Ecuador, entregando inteligencia competitiva
accionable para equipos de Pricing y Producto.

**Transforma 8 horas de trabajo manual en 10 minutos automatizados.**

| Metrica | Valor |
|---|---|
| ISPs monitoreados | 8 (Netlife, Claro, Xtrim, CNT, Ecuanet, Fibramax, Alfanet, Celerity) |
| Columnas de datos | 42 (30 base + 10 KPIs + 2 ML) |
| Proveedores LLM | 5 (DeepSeek, Gemini, Groq, Mistral, OpenAI) |
| Tests | 85 pasando |
| Precision objetivo | >95% |

---

## Quick Start

### Opcion A: Con Docker (Recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/FerQode/benchmark-360.git
cd benchmark-360

# 2. Configurar API keys
cp .env.example .env
# Editar .env con tus API keys

# 3. Ejecutar con Docker Compose
docker compose up --build

# 4. Los resultados estaran en:
#    - data/output/benchmark_industria.parquet
#    - data/output/charts/*.png
#    - benchmark_industria_notebook.ipynb
```

### Opcion B: Sin Docker (Desarrollo local)

```bash
# 1. Clonar el repositorio
git clone https://github.com/FerQode/benchmark-360.git
cd benchmark-360

# 2. Instalar uv (gestor de paquetes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Instalar dependencias
uv pip install --system -r pyproject.toml
playwright install chromium --with-deps

# 4. Configurar API keys
cp .env.example .env
# Editar .env con tus API keys

# 5. Ejecutar el pipeline
python run_pipeline.py

# 6. Aplicar capa ML
python scripts/enrich_and_export.py

# 7. Generar notebook
python scripts/generate_notebook.py

# 8. Ejecutar tests
python -m pytest tests/ -v
```

### Opcion C: Solo datos demo (sin API keys)

```bash
# Genera un dataset sintetico realista para explorar
python scripts/build_demo_dataset.py
python scripts/enrich_and_export.py
python scripts/generate_notebook.py
```

---

## Arquitectura

```
INTERNET (8 ISPs)
     |
     v
+-------------------+     +-------------------+     +-------------------+
| CAPA 1            |     | CAPA 2            |     | CAPA 3            |
| Extraccion        | --> | Inteligencia IA   | --> | Validacion        |
| Playwright+httpx  |     | LLM Waterfall x5  |     | Pydantic V2       |
| Semantic Tiling   |     | Guardrails        |     | snake_case        |
| Cookie Handler    |     | SHA-256 Cache     |     | Parquet+Snappy    |
+-------------------+     +-------------------+     +-------------------+
                                                           |
                                                           v
                                                    +-------------------+
                                                    | CAPA 4            |
                                                    | Machine Learning  |
                                                    | KMeans + KNN      |
                                                    | Alertas           |
                                                    +-------------------+
```

### Patrones de Diseno Implementados

| Patron | Uso | Archivo |
|---|---|---|
| Strategy | Cada ISP tiene su propia estrategia de scraping | `src/scrapers/base_scraper.py` |
| Singleton | Pool de conexiones LLM reutilizable | `src/processors/llm_client_factory.py` |
| Circuit Breaker | Desactivar proveedores LLM que fallan | `src/processors/provider_registry.py` |
| Factory | Crear clientes LLM bajo demanda | `src/processors/llm_client_factory.py` |
| Registry | Configuracion centralizada de ISPs | `src/scrapers/isp_registry.py` |
| 12-Factor App | Configuracion desde entorno | `src/config/settings.py` |

---

## Estructura del Proyecto

```
benchmark_360/
├── .env.example                    # Template de variables de entorno
├── .github/workflows/
│   ├── daily_benchmark.yml         # Pipeline diario automatizado
│   └── tests.yml                   # CI en cada push/PR
├── Dockerfile                      # Container de produccion
├── docker-compose.yml              # Orquestacion local
├── pyproject.toml                  # Dependencias (uv)
├── run_pipeline.py                 # Punto de entrada principal
├── src/
│   ├── config/settings.py          # Configuracion centralizada
│   ├── models/plan_model.py        # Schema Pydantic V2 (30+ cols)
│   ├── scrapers/
│   │   ├── base_scraper.py         # Scraper base + Semantic Tiling
│   │   ├── isp_registry.py         # Registry de 8 ISPs
│   │   ├── cookie_handler.py       # Dismiss-and-Verify
│   │   └── tc_scraper.py           # Terminos y Condiciones
│   ├── processors/
│   │   ├── multi_provider_adapter.py  # LLM Waterfall (5 proveedores)
│   │   ├── guardrails.py           # Anti Prompt Injection
│   │   ├── llm_cache.py            # Cache SHA-256
│   │   ├── strategic_labels.py     # 10 KPIs estrategicos
│   │   ├── market_clustering.py    # KMeans + KNN
│   │   └── competitive_alerts.py   # Motor de alertas
│   └── utils/
│       ├── errors.py               # Excepciones personalizadas
│       ├── logger.py               # Loguru estructurado
│       └── robots_checker.py       # Compliance etico
├── scripts/
│   ├── build_demo_dataset.py       # Generador de datos sinteticos
│   ├── enrich_and_export.py        # Pipeline ML
│   └── generate_notebook.py        # Generador de notebook
├── tests/                          # 85 tests
└── data/
    ├── output/                     # Parquet + charts
    └── cache/                      # Cache LLM
```

---

## Variables de Entorno

Crear un archivo `.env` basado en `.env.example`:

```bash
# API Keys (al menos una es necesaria)
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=AI...
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...
OPENAI_API_KEY=sk-...

# Configuracion del Pipeline
CACHE_TTL_HOURS=24
MAX_CONCURRENT_SCRAPERS=3
REQUEST_DELAY_MIN=2.0
REQUEST_DELAY_MAX=5.0
```

---

## Tests

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Solo tests de ML
python -m pytest tests/test_ml_processors.py -v

# Con cobertura
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Roadmap: Produccion en AWS

El proyecto esta preparado para desplegarse en AWS con un costo estimado
de **~$4.10/mes** usando ECS Fargate + S3 + Athena.

Consultar `docs/aws_roadmap.md` para el plan detallado de migracion.

---

## Equipo

**Fernando Quinapallo** — DataCrafters / Hackathon Interact2Hack / Netlife 2026
- 💻 **Repositorio Público:** [https://github.com/FerQode/benchmark-360](https://github.com/FerQode/benchmark-360) (¡Te invitamos a revisar detenidamente nuestra forma de trabajo y código fuente!)
- 🔗 **LinkedIn:** [www.linkedin.com/in/fernando-quinapallo-dev](https://www.linkedin.com/in/fernando-quinapallo-dev)

---

## Licencia

MIT License. Ver `LICENSE` para mas detalles.
