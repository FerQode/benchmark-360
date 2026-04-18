# src/pipeline/__init__.py
"""
Pipeline package — Orchestrator and Parquet writer for Benchmark 360.

Typical usage example:
    >>> from src.pipeline import PipelineOrchestrator, ParquetWriter
    >>> orchestrator = PipelineOrchestrator.from_env()
    >>> report = await orchestrator.run()
"""

from src.pipeline.orchestrator import (
    ISPPipelineResult,
    PipelineOrchestrator,
    PipelineReport,
)
from src.pipeline.parquet_writer import PARQUET_SCHEMA, ParquetWriter

__all__ = [
    "PipelineOrchestrator",
    "PipelineReport",
    "ISPPipelineResult",
    "ParquetWriter",
    "PARQUET_SCHEMA",
]
