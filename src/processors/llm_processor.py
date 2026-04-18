# src/processors/llm_processor.py
"""
LLM Processor — Text-based plan extraction with dual-provider fallback.

Orchestrates the full text extraction pipeline for a single ISP:
  1. Sanitize content through GuardrailsEngine
  2. Chunk large pages to stay within token limits
  3. Call primary LLM with versioned prompts (OpenAI or Gemini)
  4. On RateLimitError/APIError: automatically switch to fallback
  5. Validate and parse JSON output via guardrails
  6. Return structured LLMExtractionResult for Phase 6 normalizer

Typical usage example:
    >>> factory = LLMClientFactory()
    >>> processor = LLMProcessor(
    ...     primary_client=factory.get_primary_client(),
    ...     fallback_client=factory.get_fallback_client(),
    ...     guardrails=GuardrailsEngine(),
    ... )
    >>> result = await processor.extract_plans(scraped_page, company_info)
    >>> print(f"Extracted {len(result.raw_plans)} plans")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger
from openai import APIError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.processors.guardrails import GuardrailsEngine
from src.processors.llm_client_factory import LLMClient
from src.processors.prompts import (
    PROMPT_VERSION,
    build_text_extraction_messages,
)
from src.scrapers.base_scraper import ScrapedPage
from src.utils.company_registry import CompanyInfo

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

MAX_TOKENS_RESPONSE: int = 4_096
CHUNK_SIZE_CHARS: int = 70_000
MIN_CONTENT_LENGTH: int = 200


# ─────────────────────────────────────────────────────────────────
# Result DTO
# ─────────────────────────────────────────────────────────────────


@dataclass
class LLMExtractionResult:
    """Result of LLM text extraction for one ISP page.

    Attributes:
        isp_key: Source ISP identifier.
        raw_plans: Raw plan dicts from LLM — not yet Pydantic validated.
            These go directly into the Phase 6 normalizer.
        arma_tu_plan_config: Configurable plan config dict if detected.
            Passed to ArmaTuPlanHandler for Cartesian expansion.
        chunks_processed: Successfully processed chunk count.
        chunks_failed: Failed chunk count.
        prompt_version: Prompt template version for traceability.
        model_used: Actual model (may differ from primary if fallback).
        provider_used: "openai" or "gemini" — whichever succeeded.
        fallback_activated: True if fallback provider was used.
        total_llm_calls: Total API calls made (for cost tracking).
        extracted_at: UTC timestamp of completion.
        errors: Non-fatal error messages accumulated during extraction.
    """

    isp_key: str
    raw_plans: list[dict] = field(default_factory=list)
    arma_tu_plan_config: dict | None = None
    chunks_processed: int = 0
    chunks_failed: int = 0
    prompt_version: str = PROMPT_VERSION
    model_used: str = ""
    provider_used: str = ""
    fallback_activated: bool = False
    total_llm_calls: int = 0
    extracted_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Ratio of successfully processed chunks (0.0 to 1.0).

        Returns:
            Float between 0.0 and 1.0. Returns 0.0 if no chunks ran.
        """
        total = self.chunks_processed + self.chunks_failed
        return self.chunks_processed / total if total > 0 else 0.0

    @property
    def has_results(self) -> bool:
        """True if at least one plan or configurable plan was found."""
        return bool(self.raw_plans) or self.arma_tu_plan_config is not None


# ─────────────────────────────────────────────────────────────────
# LLMProcessor
# ─────────────────────────────────────────────────────────────────


class LLMProcessor:
    """GPT-4o-mini / Gemini 2.5 Flash text extraction with fallback.

    If the primary LLM client (OpenAI or Gemini) raises a
    RateLimitError or APIError, the processor automatically retries
    the failed chunk using the fallback client.

    Args:
        primary_client: Primary LLMClient from LLMClientFactory.
        fallback_client: Optional fallback LLMClient. None disables
            automatic provider switching.
        guardrails: GuardrailsEngine for sanitization and JSON validation.
        temperature: LLM temperature. 0.0 for deterministic JSON output.

    Example:
        >>> processor = LLMProcessor(
        ...     primary_client=factory.get_primary_client(),
        ...     fallback_client=factory.get_fallback_client(),
        ...     guardrails=GuardrailsEngine(),
        ... )
        >>> result = await processor.extract_plans(page, company_info)
    """

    def __init__(
        self,
        primary_client: LLMClient,
        fallback_client: LLMClient | None,
        guardrails: GuardrailsEngine,
        temperature: float = 0.0,
    ) -> None:
        self._primary = primary_client
        self._fallback = fallback_client
        self._guardrails = guardrails
        self._temperature = temperature

    async def extract_plans(
        self,
        scraped_page: ScrapedPage,
        company_info: CompanyInfo,
    ) -> LLMExtractionResult:
        """Extract all internet plans from a scraped ISP page via text.

        Pipeline:
        1. Guardrail inspection + sanitization of raw content
        2. Chunk content to fit model token limits
        3. Per-chunk LLM call with primary → fallback routing
        4. Merge all chunk results into a single result object

        Args:
            scraped_page: ScrapedPage DTO from the scraping layer.
            company_info: CompanyInfo with empresa and marca names.

        Returns:
            LLMExtractionResult with raw_plans list and metadata.
        """
        result = LLMExtractionResult(
            isp_key=scraped_page.isp_key,
            model_used=self._primary.text_model,
            provider_used=self._primary.provider_name,
        )

        content = scraped_page.text_content or scraped_page.html_content

        if len(content) < MIN_CONTENT_LENGTH:
            msg = (
                f"Content too short ({len(content)} chars)"
                " — skipping LLM text extraction"
            )
            logger.warning("[{}] {}", scraped_page.isp_key, msg)
            result.errors.append(msg)
            return result

        # ── Step 1: Guardrail inspection ──────────────────────────
        guard = self._guardrails.inspect(content)
        safe_content = guard.sanitized_text

        if not guard.is_safe:
            logger.warning(
                "[{}] ⚠️ Guardrail risk={} score={} — using sanitized",
                scraped_page.isp_key,
                guard.risk_level.name,
                guard.risk_score,
            )

        # ── Step 2: Chunk content ─────────────────────────────────
        chunks = self._chunk_content(safe_content)
        total_chunks = len(chunks)

        logger.info(
            "[{}] 🤖 Text extraction — {} chunk(s), provider={}",
            scraped_page.isp_key,
            total_chunks,
            self._primary.provider_name,
        )

        # ── Step 3: Per-chunk extraction ──────────────────────────
        for idx, chunk in enumerate(chunks):
            messages = build_text_extraction_messages(
                isp_key=scraped_page.isp_key,
                marca=company_info.marca,
                empresa=company_info.empresa,
                text_content=chunk,
                chunk_index=idx,
                total_chunks=total_chunks,
            )
            await self._process_chunk(
                messages=messages,
                chunk_idx=idx,
                total_chunks=total_chunks,
                result=result,
                isp_key=scraped_page.isp_key,
            )

        logger.info(
            "[{}] 🏁 Text done — {} plans, {}/{} OK, "
            "provider={}, fallback={}",
            scraped_page.isp_key,
            len(result.raw_plans),
            result.chunks_processed,
            total_chunks,
            result.provider_used,
            result.fallback_activated,
        )
        return result

    # ── Private ───────────────────────────────────────────────────

    async def _process_chunk(
        self,
        messages: list[dict],
        chunk_idx: int,
        total_chunks: int,
        result: LLMExtractionResult,
        isp_key: str,
    ) -> None:
        """Process one content chunk with primary → fallback routing.

        Tries the primary client first. On RateLimitError or APIError,
        retries with the fallback if available.

        Args:
            messages: Prepared OpenAI-format messages for this chunk.
            chunk_idx: Zero-based chunk index.
            total_chunks: Total chunks for log context.
            result: Mutable result to accumulate extracted plans into.
            isp_key: ISP key for log context.
        """
        clients: list[LLMClient] = [self._primary]
        if self._fallback:
            clients.append(self._fallback)

        for attempt, llm_client in enumerate(clients):
            is_fallback = attempt > 0
            try:
                raw = await self._call_with_retry(
                    client=llm_client.client,
                    model=llm_client.text_model,
                    messages=messages,
                    temperature=self._temperature,
                )
                result.total_llm_calls += 1

                if is_fallback:
                    result.fallback_activated = True
                    result.model_used = llm_client.text_model
                    result.provider_used = llm_client.provider_name
                    logger.info(
                        "[{}] ⚡ Fallback {} succeeded chunk {}",
                        isp_key,
                        llm_client.provider_name,
                        chunk_idx,
                    )

                is_valid, parsed = self._guardrails.validate_llm_output(raw)

                if not is_valid:
                    msg = (
                        f"Chunk {chunk_idx}: invalid JSON "
                        f"(provider={llm_client.provider_name})"
                    )
                    logger.error("[{}] {}", isp_key, msg)
                    result.errors.append(msg)
                    result.chunks_failed += 1
                    return

                plans = parsed.get("plans", [])
                result.raw_plans.extend(plans)

                if (
                    parsed.get("arma_tu_plan_config")
                    and result.arma_tu_plan_config is None
                ):
                    result.arma_tu_plan_config = parsed["arma_tu_plan_config"]

                result.chunks_processed += 1
                meta = parsed.get("extraction_metadata", {})
                logger.info(
                    "[{}] ✅ Chunk {}/{} → {} plans "
                    "(conf={:.0%}, provider={})",
                    isp_key,
                    chunk_idx + 1,
                    total_chunks,
                    len(plans),
                    meta.get("confidence", 0),
                    llm_client.provider_name,
                )
                return  # success — stop trying

            except (RateLimitError, APIError) as exc:
                if is_fallback or not self._fallback:
                    msg = f"Chunk {chunk_idx}: all providers failed — {exc}"
                    logger.error("[{}] ❌ {}", isp_key, msg)
                    result.errors.append(msg)
                    result.chunks_failed += 1
                    return
                logger.warning(
                    "[{}] ⚠️ Primary {} failed chunk {} → "
                    "switching to {} — {}",
                    isp_key,
                    llm_client.provider_name,
                    chunk_idx,
                    self._fallback.provider_name,
                    exc,
                )

            except Exception as exc:
                msg = f"Chunk {chunk_idx}: unexpected error — {exc}"
                logger.error("[{}] ❌ {}", isp_key, msg)
                result.errors.append(msg)
                result.chunks_failed += 1
                return

    @staticmethod
    async def _call_with_retry(
        client: AsyncOpenAI,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
    ) -> str:
        """Call any AsyncOpenAI-compatible client with retry backoff.

        Args:
            client: AsyncOpenAI instance (OpenAI or Gemini compat).
            model: Model identifier string.
            messages: Prepared OpenAI-format message list.
            temperature: Sampling temperature.

        Returns:
            Raw string content from the LLM response.

        Raises:
            RateLimitError: After 3 failed attempts.
            APIError: After 3 failed attempts.
        """
        @retry(
            retry=retry_if_exception_type((RateLimitError, APIError)),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=5, max=30),
            reraise=True,
        )
        async def _inner() -> str:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS_RESPONSE,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            logger.debug(
                "LLM: model={} len={} finish={}",
                model,
                len(content),
                response.choices[0].finish_reason,
            )
            return content

        return await _inner()

    @staticmethod
    def _chunk_content(text: str) -> list[str]:
        """Split content into chunks at paragraph boundaries.

        Args:
            text: Full sanitized text to split.

        Returns:
            List of chunks each <= CHUNK_SIZE_CHARS characters.
        """
        if len(text) <= CHUNK_SIZE_CHARS:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE_CHARS
            if end >= len(text):
                chunks.append(text[start:])
                break
            split_pos = text.rfind("\n\n", start, end)
            if split_pos == -1:
                split_pos = text.rfind("\n", start, end)
            if split_pos == -1:
                split_pos = end
            chunks.append(text[start:split_pos])
            start = split_pos + 1

        logger.debug(
            "Chunked {} chars into {} parts", len(text), len(chunks)
        )
        return chunks
