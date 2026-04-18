# src/processors/llm_client_factory.py
"""
LLM Client Factory — Centralized dual-provider client management.

Implements the Factory pattern for creating AsyncOpenAI-compatible
clients for both OpenAI and Google Gemini providers.

Key architectural decision:
    Both providers use the same OpenAI SDK. Gemini is accessed via
    its native OpenAI-compatible endpoint:
    https://generativelanguage.googleapis.com/v1beta/openai/

    This avoids duplicating prompt logic and maintains a single
    message format across providers.

Environment variables consumed:
    PRIMARY_LLM_PROVIDER: "openai" | "gemini" | "auto"
    OPENAI_API_KEY: OpenAI secret key.
    OPENAI_TEXT_MODEL: Text model identifier.
    OPENAI_VISION_MODEL: Vision model identifier.
    GEMINI_API_KEY: Google AI Studio API key.
    GEMINI_TEXT_MODEL: Gemini text model identifier.
    GEMINI_VISION_MODEL: Gemini vision model identifier.

Typical usage example:
    >>> factory = LLMClientFactory()
    >>> primary = factory.get_primary_client()
    >>> fallback = factory.get_fallback_client()
    >>> print(primary.provider_name)
    'openai'
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from loguru import logger
from openai import AsyncOpenAI

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

_OPENAI_BASE_URL: str = "https://api.openai.com/v1"
_GEMINI_COMPAT_BASE_URL: str = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)


# ─────────────────────────────────────────────────────────────────
# LLMClient — thin wrapper around AsyncOpenAI
# ─────────────────────────────────────────────────────────────────


@dataclass
class LLMClient:
    """Thin wrapper around AsyncOpenAI with provider metadata.

    Carries provider name and model names alongside the AsyncOpenAI
    client so callers always know which provider they are using.

    Attributes:
        client: AsyncOpenAI instance pointing to OpenAI API or
            Gemini's compatibility endpoint.
        provider_name: Human-readable identifier: "openai" or "gemini".
        text_model: Model identifier for text-based extraction.
        vision_model: Model identifier for vision-based extraction.
    """

    client: AsyncOpenAI
    provider_name: str
    text_model: str
    vision_model: str

    def __repr__(self) -> str:
        """Return debug-friendly string representation."""
        return (
            f"LLMClient(provider={self.provider_name!r}, "
            f"text={self.text_model!r}, "
            f"vision={self.vision_model!r})"
        )


# ─────────────────────────────────────────────────────────────────
# LLMClientFactory
# ─────────────────────────────────────────────────────────────────


class LLMClientFactory:
    """Factory for creating OpenAI-compatible LLM clients.

    Reads configuration from environment variables and creates
    LLMClient instances for OpenAI and/or Gemini providers.

    In "auto" mode, prefers OpenAI if the key is available and
    returns Gemini as the fallback provider.

    Args:
        primary_provider: Override PRIMARY_LLM_PROVIDER env var.
            One of: "openai", "gemini", "auto".

    Example:
        >>> factory = LLMClientFactory()
        >>> primary = factory.get_primary_client()
        >>> fallback = factory.get_fallback_client()
        >>> config = factory.validate_configuration()
        >>> print(config)
        {'openai': True, 'gemini': True}
    """

    def __init__(self, primary_provider: str | None = None) -> None:
        self._primary_provider: str = (
            (primary_provider or os.getenv("PRIMARY_LLM_PROVIDER", "auto"))
            .lower()
            .strip()
        )

        self._openai_key: str | None = os.getenv("OPENAI_API_KEY")
        self._gemini_key: str | None = os.getenv("GEMINI_API_KEY")

        self._openai_text_model: str = os.getenv(
            "OPENAI_TEXT_MODEL", "gpt-4o-mini"
        )
        self._openai_vision_model: str = os.getenv(
            "OPENAI_VISION_MODEL", "gpt-4o"
        )
        self._gemini_text_model: str = os.getenv(
            "GEMINI_TEXT_MODEL", "gemini-2.5-flash"
        )
        self._gemini_vision_model: str = os.getenv(
            "GEMINI_VISION_MODEL", "gemini-2.5-flash"
        )

        logger.info(
            "LLMClientFactory — provider={}, openai={}, gemini={}",
            self._primary_provider,
            "set" if self._openai_key else "MISSING",
            "set" if self._gemini_key else "MISSING",
        )

    def get_primary_client(self) -> LLMClient:
        """Create and return the configured primary LLM client.

        In 'auto' mode: prefers OpenAI if key available,
        otherwise uses Gemini.

        Returns:
            LLMClient configured for the primary provider.

        Raises:
            RuntimeError: If no API keys are configured.
        """
        if self._primary_provider == "openai":
            return self._build_openai_client()

        if self._primary_provider == "gemini":
            return self._build_gemini_client()

        # auto mode
        if self._openai_key:
            logger.info("Auto mode: OpenAI key found → primary=openai")
            return self._build_openai_client()

        if self._gemini_key:
            logger.info("Auto mode: no OpenAI key → primary=gemini")
            return self._build_gemini_client()

        raise RuntimeError(
            "No LLM API keys configured. "
            "Set OPENAI_API_KEY or GEMINI_API_KEY in your .env file."
        )

    def get_fallback_client(self) -> LLMClient | None:
        """Create and return the fallback LLM client.

        Returns the alternative provider to the primary. Returns None
        if only one provider is configured.

        Returns:
            LLMClient for the fallback provider, or None.
        """
        if self._primary_provider == "openai" and self._gemini_key:
            logger.info("Fallback client: gemini (primary=openai)")
            return self._build_gemini_client()

        if self._primary_provider == "gemini" and self._openai_key:
            logger.info("Fallback client: openai (primary=gemini)")
            return self._build_openai_client()

        if self._primary_provider == "auto":
            if self._openai_key and self._gemini_key:
                logger.info("Auto fallback: gemini")
                return self._build_gemini_client()

        logger.warning(
            "No fallback client available — only one provider key configured"
        )
        return None

    def validate_configuration(self) -> dict[str, bool]:
        """Check which providers are properly configured.

        Returns:
            Dict mapping provider names to availability booleans.

        Example:
            >>> factory.validate_configuration()
            {'openai': True, 'gemini': False}
        """
        return {
            "openai": bool(self._openai_key),
            "gemini": bool(self._gemini_key),
        }

    # ── Private builders ──────────────────────────────────────────

    def _build_openai_client(self) -> LLMClient:
        """Instantiate LLMClient for OpenAI.

        Returns:
            LLMClient with AsyncOpenAI pointing to OpenAI API.

        Raises:
            RuntimeError: If OPENAI_API_KEY is not set.
        """
        if not self._openai_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Cannot build OpenAI client."
            )
        return LLMClient(
            client=AsyncOpenAI(
                api_key=self._openai_key,
                base_url=_OPENAI_BASE_URL,
            ),
            provider_name="openai",
            text_model=self._openai_text_model,
            vision_model=self._openai_vision_model,
        )

    def _build_gemini_client(self) -> LLMClient:
        """Instantiate LLMClient for Gemini via OpenAI compat endpoint.

        Uses Google's native OpenAI-compatible endpoint so the same
        AsyncOpenAI SDK and message format works for both providers.

        Returns:
            LLMClient with AsyncOpenAI pointing to Gemini compat API.

        Raises:
            RuntimeError: If GEMINI_API_KEY is not set.
        """
        if not self._gemini_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Cannot build Gemini client."
            )
        return LLMClient(
            client=AsyncOpenAI(
                api_key=self._gemini_key,
                base_url=_GEMINI_COMPAT_BASE_URL,
            ),
            provider_name="gemini",
            text_model=self._gemini_text_model,
            vision_model=self._gemini_vision_model,
        )
