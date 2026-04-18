# Benchmark 360 🚀

**Pipeline de Inteligencia Competitiva para ISPs en Ecuador**

Benchmark 360 es una plataforma automatizada (ETL/Pipeline) de extracción, estandarización y análisis de planes de internet de los principales proveedores de telecomunicaciones en Ecuador. Utiliza Scraping asíncrono, Modelos de Lenguaje Grande (LLMs) para extracción de datos no estructurados y validaciones estrictas mediante Pydantic.

## 🌟 Características Principales

- **Scraping Resiliente y Concurrente:** Arquitectura de 3 niveles (HTTPX rápido, Playwright para JS rendering, y capturas de pantalla para datos en imágenes).
- **Extracción de Datos Híbrida (Texto + Visión):** Utiliza LLMs (`GPT-4o-mini`, `Gemini 2.5 Flash`) para leer texto HTML y/o analizar capturas de pantalla promocionales con un formato de salida JSON estructurado.
- **Tolerancia a Fallos (Dual-Provider Fallback):** Si OpenAI agota su cuota o tiene problemas de red, el sistema cambia automáticamente a Gemini sin interrumpir la orquestación.
- **Normalización Inteligente:** Estandarización de precios, velocidades, IVA y catálogos de Productos y Servicios Adicionales (PYS).
- **Persistencia en Parquet:** Salida estructurada de alta eficiencia ideal para Data Lakes, Pandas y análisis en Jupyter Notebooks.
- **Seguridad y Ética:** Cumplimiento de `robots.txt` y sanitización del contenido (Guardrails) antes de pasarlo a los LLMs para prevenir Prompt Injections.

## 🏗️ Arquitectura en Fases

El proyecto ha sido construido de forma iterativa y modular mediante Clean Architecture:

*   **Fase 0 & 1:** Fundamentos, Arquitectura de Decisión y Setup del Proyecto (`uv`).
*   **Fase 2:** Capa de Modelos - Esquemas y validación estricta usando `Pydantic V2`.
*   **Fase 3:** Capa de Seguridad - Motor de `Guardrails` y `RobotsChecker`.
*   **Fase 4:** Capa de Scraping - Patrón Factory con 8 scrapers dedicados.
*   **Fase 5:** Capa de Procesamiento LLM - Prompts as Code, extracción de texto y visión.
*   **Fase 6:** Capa de Normalización - `PlanNormalizer` y catálogo `PYS`.
*   **Fase 7:** Pipeline Orchestrator - Ejecución concurrente asíncrona (`asyncio`).
*   **Fase 8:** Capa de Datos - Escritor Parquet y validación de calidad.
*   *Fase 9 al 11 (Próximamente):* Testing E2E, Notebooks analíticos y Storytelling.

## 📡 Proveedores Integrados (ISPs)

- Netlife
- Claro
- CNT
- Xtrim
- Fibramax
- Puntonet (Celerity)
- Ecuanet
- Alfanet

---

## 🛠️ Instalación y Configuración

El proyecto utiliza [uv](https://github.com/astral-sh/uv) como gestor rápido de dependencias.

1. **Clonar y preparar entorno:**
   ```bash
   git clone https://github.com/FerQode/benchmark-360.git
   cd benchmark_360
   uv venv
   uv pip install -e .
   ```

2. **Instalar navegadores para Playwright:**
   ```bash
   uv run playwright install chromium
   ```

3. **Configuración del Entorno (.env):**
   Copia el archivo de ejemplo y configura tus API Keys.
   ```bash
   cp .env.example .env
   ```
   **Contenido clave del `.env`:**
   ```env
   OPENAI_API_KEY=sk-...
   GEMINI_API_KEY=AIza...
   PRIMARY_LLM_PROVIDER=auto  # auto, openai, o gemini
   DRY_RUN=false              # false para usar LLMs en producción
   ```

---

## 🚀 Uso del Pipeline

El punto de entrada principal es `scripts/run_pipeline.py`. 
Para evitar problemas de codificación de caracteres en consola Windows, se recomienda fijar `PYTHONIOENCODING`.

### Ejecución en Producción (Todos los ISPs)
Extrae datos reales utilizando créditos de LLM y guarda un archivo Parquet.
```bash
$env:PYTHONIOENCODING="utf-8"
uv run python scripts/run_pipeline.py --report
```

### Ejecución de ISPs Específicos
Si solo deseas analizar proveedores particulares:
```bash
uv run python scripts/run_pipeline.py --isps netlife claro xtrim --report
```

### Modo Prueba (Dry Run)
Prueba la infraestructura de scraping sin consumir créditos de API ni llamar a los LLMs (ideal para probar los selectores de Playwright).
```bash
uv run python scripts/run_pipeline.py --isps netlife --dry-run
```

---

## 🧪 Testing

La suite de pruebas contiene más de 230 tests unitarios con una alta cobertura en las capas centrales.

```bash
uv run pytest tests/unit/ -v --tb=short
```

---

*Desarrollado para proveer inteligencia competitiva accionable con IA generativa.*
