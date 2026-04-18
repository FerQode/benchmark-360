# scripts/run_pipeline.py
"""
CLI entry point for the Benchmark 360 pipeline.

PRODUCTION (extracts real data with LLMs):
    uv run python scripts/run_pipeline.py

SPECIFIC ISPs:
    uv run python scripts/run_pipeline.py --isps netlife cnt claro

DRY RUN (scraping only, NO LLM, NO Parquet — for scraper testing):
    uv run python scripts/run_pipeline.py --dry-run

VIEW RESULTS:
    uv run python scripts/run_pipeline.py --report
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import io

if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline.orchestrator import PipelineOrchestrator
from src.scrapers import ALL_ISP_URLS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark 360 — ISP Competitive Intelligence Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  Production (LLMs ON):
    uv run python scripts/run_pipeline.py

  Test scraping only (LLMs OFF):
    uv run python scripts/run_pipeline.py --dry-run

  Specific ISPs:
    uv run python scripts/run_pipeline.py --isps netlife claro xtrim
        """,
    )
    parser.add_argument(
        "--isps",
        nargs="+",
        choices=list(ALL_ISP_URLS.keys()),
        default=None,
        help="ISP keys to process (default: all 8)",
        metavar="ISP",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Scrape only — NO LLM calls, NO Parquet write. "
            "Use to test scrapers without consuming API credits."
        ),
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print DataFrame preview after run",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=3,
        help="Max concurrent ISPs (default: 3)",
    )
    return parser.parse_args()


async def main() -> int:
    """Run pipeline and return exit code (0=success, 1=no plans)."""
    args = parse_args()

    # Build orchestrator from .env
    orchestrator = PipelineOrchestrator.from_env()

    # CLI flags override .env
    if args.dry_run:
        orchestrator._dry_run = True
        print(
            "\n⚠️  DRY RUN MODE: LLM calls disabled. "
            "Remove --dry-run for real data extraction.\n"
        )

    orchestrator._semaphore = asyncio.Semaphore(args.concurrent)

    # Run pipeline
    report = await orchestrator.run(isp_keys=args.isps)
    report.print_summary()

    # Optional DataFrame preview
    if args.report and report.output_path and report.output_path.exists():
        from src.pipeline.parquet_writer import ParquetWriter

        writer = ParquetWriter(
            output_dir=report.output_path.parent
        )
        df = writer.read(path=report.output_path)

        print("── DataFrame Preview (first 10 rows) ───────────────")
        preview_cols = [
            "marca", "nombre_plan",
            "velocidad_download_mbps", "precio_plan",
            "pys_adicionales", "tecnologia",
        ]
        available = [c for c in preview_cols if c in df.columns]
        print(df[available].head(10).to_string(index=False))
        print(f"\nTotal rows: {len(df)} | Columns: {len(df.columns)}")
        print(f"ISPs: {sorted(df['marca'].dropna().unique().tolist())}")

    elif args.report and not report.output_path:
        print(
            "\n⚠️  No Parquet file to preview "
            "(dry_run=True or no plans extracted)"
        )

    return 0 if report.total_plans > 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
