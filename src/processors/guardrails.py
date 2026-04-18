# src/processors/guardrails.py
"""
Guardrails engine to protect against injection attacks.

Implements a 4-layer defense mechanism to validate both
inputs before sending to LLM and outputs from the LLM.
Calculates a quantitative risk score for each inspection.
"""

import enum
import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger


class RiskLevel(enum.Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GuardrailResult:
    """Result of a guardrail inspection."""
    is_safe: bool
    risk_score: int
    risk_level: RiskLevel
    detected_signatures: list[str] = field(default_factory=list)
    sanitized_text: str = ""
    inspected_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


class GuardrailsEngine:
    """4-layer defense engine for text validation.
    
    Layers:
    1. HTML/Entity sanitization
    2. SQL Injection detection
    3. XSS detection
    4. Prompt Injection detection
    """

    # 27 injection signatures combined into categories
    _SIGNATURES = {
        "sql_injection": [
            r"(?i)\bUNION\b\s+ALL\b", r"(?i)\bSELECT\b\s+.*\bFROM\b",
            r"(?i)\bDROP\b\s+TABLE\b", r"(?i)\bINSERT\b\s+INTO\b",
            r"(?i)\bDELETE\b\s+FROM\b", r"(?i)\bUPDATE\b\s+.*\bSET\b",
            r"(?i)\bTRUNCATE\b\s+TABLE\b", r"(?i)\bALTER\b\s+TABLE\b",
            r"(?i)'\s*OR\s*'1'\s*=\s*'1", r"(?i)\"\s*OR\s*\"1\"\s*=\s*\"1",
        ],
        "xss_injection": [
            r"(?i)<script[^>]*>.*?</script>", r"(?i)javascript:",
            r"(?i)onerror=", r"(?i)onload=", r"(?i)eval\(",
            r"(?i)alert\(", r"(?i)document\.cookie", r"(?i)window\.location",
        ],
        "prompt_injection": [
            r"(?i)ignore\s+(all\s+)?(previous\s+)?instructions",
            r"(?i)disregard\s+(all\s+)?(previous\s+)?instructions",
            r"(?i)forget\s+(all\s+)?(previous\s+)?instructions",
            r"(?i)you\s+are\s+now", r"(?i)roleplay\s+as",
            r"(?i)translate\s+the\s+following", r"(?i)system\s+prompt",
            r"(?i)bypass", r"(?i)jailbreak",
        ]
    }

    def inspect(self, text: str) -> GuardrailResult:
        """Run all defense layers. Score is capped per category (not per match)."""
        sanitized = self._sanitize(text)
        
        score = 0
        detected: set[str] = set()
        
        # Test original text for signatures so we score based on what was there
        for category, patterns in self._SIGNATURES.items():
            category_hit = False
            for pattern in patterns:
                # Need to test on html.unescaped to find encoded attacks
                if re.search(pattern, html.unescape(text)):
                    if not category_hit:
                        score += 10
                        category_hit = True
                    detected.add(category)
                    
        # Calculate risk level
        if score == 0:
            level = RiskLevel.SAFE
        elif score < 20:
            level = RiskLevel.LOW
        elif score < 40:
            level = RiskLevel.MEDIUM
        elif score < 60:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL
            
        result = GuardrailResult(
            is_safe=(level in [RiskLevel.SAFE, RiskLevel.LOW]),
            risk_score=score,
            risk_level=level,
            detected_signatures=list(detected),
            sanitized_text=sanitized
        )
        
        if result.is_safe:
            logger.info("Guardrail check passed with SAFE/LOW risk score: {}", score)
        else:
            logger.warning("Guardrail check failed! Risk Level: {}, Score: {}", level.name, score)
            
        return result

    def _sanitize(self, text: str) -> str:
        """Layer 1: Unescape HTML entities and strip dangerous content.

        Converts HTML entities to their characters first (so encoded
        payloads are detected by regex patterns), then neutralizes
        matched signatures by replacing them with a safe placeholder.

        Args:
            text: Raw input text to sanitize.

        Returns:
            Sanitized text safe for LLM submission.
        """
        # Step 1: Unescape HTML entities (catch encoded injections)
        text = html.unescape(text)

        # Step 2: Strip control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Step 3: Neutralize all signature matches (don't just detect)
        for patterns in self._SIGNATURES.values():
            for pattern in patterns:
                text = re.sub(pattern, "[REDACTED]", text)

        # Step 4: Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def validate_llm_output(self, output: str) -> tuple[bool, dict]:
        """Validate and parse LLM JSON output, handling markdown code blocks.

        Attempts direct JSON parse first, then strips markdown fences
        if the first attempt fails. This handles GPT-4o's tendency to
        wrap JSON in ```json ... ``` blocks.

        Args:
            output: Raw string response from the LLM API.

        Returns:
            Tuple of (is_valid: bool, parsed_data: dict).
            parsed_data is empty dict if validation fails.
        """
        if not output or not output.strip():
            logger.error("LLM output validation failed: empty response")
            return False, {}

        candidates = [
            output.strip(),
            # Strip markdown fences
            re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.MULTILINE),
        ]

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                logger.debug("LLM output parsed successfully ({} keys)", len(data))
                return True, data
            except json.JSONDecodeError:
                continue

        logger.error(
            "LLM output validation failed: not valid JSON even after markdown strip. "
            "First 200 chars: {}",
            output[:200],
        )
        return False, {}
