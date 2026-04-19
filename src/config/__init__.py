# src/config/__init__.py
"""Configuración centralizada del proyecto Benchmark 360."""

from src.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
