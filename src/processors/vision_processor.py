# src/processors/vision_processor.py
"""
Vision Processor — Screenshot extraction with dual-provider fallback.

Handles ISP websites that publish plans as images or banners that
are invisible to HTML/text parsers. Uses GPT-4o or Gemini 2.5 Flash
Vision via the OpenAI-compatible SDK for both providers.

Typical usage example:
    >>> processor = VisionProcessor(
    ...     primary_client=factory.get_primary_client(),
    ...     fallback_client=factory.get_fallback_client(),
    ...     guardrails=GuardrailsEngine(),
    ... )
    >>> result = await processor.extract_from_screenshots(page, info)
    >>> print(f"Vision extracted {len(result.raw_plans)} plans")
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
    build_vision_extraction_messages,
)
from src.scrapers.base_scraper import ScrapedPage
from src.utils.company_registry import CompanyInfo

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

MAX_TOKENS_VISION: int = 4_096
MAX_SCREENSHOTS_PER_BATCH: int = 5


# ─────────────────────────────────────────────────────────────────
# Result DTO
# ─────────────────────────────────────────────────────────────────


@dataclass
class VisionExtractionResult:
    """Result of GPT-4o / Gemini Vision extraction for one ISP page.

    Attributes:
        isp_key: Source ISP identifier.
        raw_plans: Raw plan dicts extracted from screenshots.
        arma_tu_plan_config: Configurable plan config if detected.
        screenshots_processed: Total screenshots analyzed successfully.
        prompt_version: Prompt version string for traceability.
        model_used: Actual model name (may differ if fallback used).
        provider_used: Provider that succeeded: "openai" or "gemini".
        fallback_activated: True if fallback provider was used.
        total_llm_calls: API call count for cost tracking.
        extracted_at: UTC timestamp of extraction completion.
        errors: Non-fatal error messages.
    """

    isp_key: str
    raw_plans: list[dict] = field(default_factory=list)
    arma_tu_plan_config: dict | None = None
    screenshots_processed: int = 0
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
    def has_results(self) -> bool:
        """True if at least one plan was extracted from screenshots."""
        return bool(self.raw_plans)


# ─────────────────────────────────────────────────────────────────
# VisionProcessor
# ─────────────────────────────────────────────────────────────────


class VisionProcessor:
    """GPT-4o / Gemini 2.5 Flash vision extraction with fallback.

    Batches screenshots into groups of MAX_SCREENSHOTS_PER_BATCH to
    minimize API calls while providing full page context. Automatically
    switches to fallback provider on RateLimitError or APIError.

    Args:
        primary_client: Primary LLMClient from LLMClientFactory.
        fallback_client: Optional fallback LLMClient.
        guardrails: GuardrailsEngine for output validation.
        temperature: LLM temperature. 0.0 for deterministic output.

    Example:
        >>> processor = VisionProcessor(
        ...     primary_client=factory.get_primary_client(),
        ...     fallback_client=factory.get_fallback_client(),
        ...     guardrails=GuardrailsEngine(),
        ... )
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

    async def extract_from_screenshots(
        self,
        scraped_page: ScrapedPage,
        company_info: CompanyInfo,
    ) -> VisionExtractionResult:
        """Extract plans from all screenshots in a ScrapedPage.

        Processes screenshots in batches and merges all results.

        Args:
            scraped_page: ScrapedPage with screenshots to analyze.
            company_info: Company legal and brand names.

        Returns:
            VisionExtractionResult with extracted raw_plans.
        """
        result = VisionExtractionResult(
            isp_key=scraped_page.isp_key,
            model_used=self._primary.vision_model,
            provider_used=self._primary.provider_name,
        )

        if not scraped_page.has_screenshots:
            logger.info(
                "[{}] No screenshots — skipping vision extraction",
                scraped_page.isp_key,
            )
            return result

        screenshots_b64 = scraped_page.screenshots_as_base64()
        batches = [
            screenshots_b64[i: i + MAX_SCREENSHOTS_PER_BATCH]
            for i in range(0, len(screenshots_b64), MAX_SCREENSHOTS_PER_BATCH)
        ]

        logger.info(
            "[{}] 👁️ Vision extraction — {} screenshot(s), "
            "{} batch(es), provider={}",
            scraped_page.isp_key,
            len(screenshots_b64),
            len(batches),
            self._primary.provider_name,
        )

        for batch_idx, batch in enumerate(batches):
            messages = build_vision_extraction_messages(
                isp_key=scraped_page.isp_key,
                marca=company_info.marca,
                empresa=company_info.empresa,
                screenshots_b64=batch,
            )
            await self._process_batch(
                messages=messages,
                batch_idx=batch_idx,
                batch_size=len(batch),
                result=result,
                isp_key=scraped_page.isp_key,
            )

        logger.info(
            "[{}] 🏁 Vision done — {} plans, {} shots, "
            "provider={}, fallback={}",
            scraped_page.isp_key,
            len(result.raw_plans),
            result.screenshots_processed,
            result.provider_used,
            result.fallback_activated,
        )
        return result

    # ── Private ───────────────────────────────────────────────────

    async def _process_batch(
        self,
        messages: list[dict],
        batch_idx: int,
        batch_size: int,
        result: VisionExtractionResult,
        isp_key: str,
    ) -> None:
        """Process one screenshot batch with primary → fallback routing.

        Args:
            messages: Multimodal messages with base64 image parts.
            batch_idx: Zero-based batch index.
            batch_size: Number of screenshots in this batch.
            result: Mutable result to accumulate plans into.
            isp_key: ISP key for log context.
        """
        clients: list[LLMClient] = [self._primary]
        if self._fallback:
            clients.append(self._fallback)

        for attempt, llm_client in enumerate(clients):
            is_fallback = attempt > 0
            try:
                raw = await self._call_vision_with_retry(
                    client=llm_client.client,
                    model=llm_client.vision_model,
                    messages=messages,
                    temperature=self._temperature,
                )
                result.total_llm_calls += 1

                if is_fallback:
                    result.fallback_activated = True
                    result.model_used = llm_client.vision_model
                    result.provider_used = llm_client.provider_name
                    logger.info(
                        "[{}] ⚡ Vision fallback {} succeeded batch {}",
                        isp_key,
                        llm_client.provider_name,
                        batch_idx,
                    )

                is_valid, parsed = self._guardrails.validate_llm_output(raw)

                if not is_valid:
                    msg = (
                        f"Vision batch {batch_idx}: invalid JSON "
                        f"(provider={llm_client.provider_name})"
                    )
                    logger.error("[{}] {}", isp_key, msg)
                    result.errors.append(msg)
                    return

                plans = parsed.get("plans", [])
                result.raw_plans.extend(plans)
                result.screenshots_processed += batch_size

                if (
                    parsed.get("arma_tu_plan_config")
                    and result.arma_tu_plan_config is None
                ):
                    result.arma_tu_plan_config = parsed["arma_tu_plan_config"]

                meta = parsed.get("extraction_metadata", {})
                logger.info(
                    "[{}] ✅ Vision batch {} → {} plans "
                    "(conf={:.0%}, provider={})",
                    isp_key,
                    batch_idx,
                    len(plans),
                    meta.get("confidence", 0),
                    llm_client.provider_name,
                )
                return

            except (RateLimitError, APIError) as exc:
                if is_fallback or not self._fallback:
                    msg = f"Vision batch {batch_idx}: all failed — {exc}"
                    logger.error("[{}] ❌ {}", isp_key, msg)
                    result.errors.append(msg)
                    return
                logger.warning(
                    "[{}] ⚠️ Vision primary {} failed batch {} → {} — {}",
                    isp_key,
                    llm_client.provider_name,
                    batch_idx,
                    self._fallback.provider_name,
                    exc,
                )

            except Exception as exc:
                msg = f"Vision batch {batch_idx}: unexpected — {exc}"
                logger.error("[{}] ❌ {}", isp_key, msg)
                result.errors.append(msg)
                return

    @staticmethod
    async def _call_vision_with_retry(
        client: AsyncOpenAI,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
    ) -> str:
        """Call Vision LLM with exponential backoff retry.

        Args:
            client: AsyncOpenAI instance (OpenAI or Gemini compat).
            model: Vision-capable model identifier.
            messages: Multimodal messages with image_url parts.
            temperature: Sampling temperature.

        Returns:
            Raw string response from the LLM.
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
                max_tokens=MAX_TOKENS_VISION,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            logger.debug(
                "Vision: model={} len={} finish={}",
                model,
                len(content),
                response.choices[0].finish_reason,
            )
            return content

        return await _inner()
