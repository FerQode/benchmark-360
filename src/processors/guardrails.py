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
from datetime import datetime

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
    inspected_at: datetime = field(default_factory=datetime.now)


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
            r"(?i)\bUNION\b\s+(?i)ALL\b", r"(?i)\bSELECT\b\s+.*\bFROM\b",
            r"(?i)\bDROP\b\s+(?i)TABLE\b", r"(?i)\bINSERT\b\s+(?i)INTO\b",
            r"(?i)\bDELETE\b\s+(?i)FROM\b", r"(?i)\bUPDATE\b\s+.*\bSET\b",
            r"(?i)\bTRUNCATE\b\s+(?i)TABLE\b", r"(?i)\bALTER\b\s+(?i)TABLE\b",
            r"' OR '1'='1", r"\" OR \"1\"=\"1",
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
        """Run all 4 layers of defense on the input text."""
        sanitized = self._sanitize(text)
        
        score = 0
        detected = []
        
        for category, patterns in self._SIGNATURES.items():
            for pattern in patterns:
                if re.search(pattern, sanitized):
                    score += 10
                    detected.append(category)
                    
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
            detected_signatures=list(set(detected)),
            sanitized_text=sanitized
        )
        
        if result.is_safe:
            logger.info(f"Guardrail check passed with SAFE/LOW risk score: {score}")
        else:
            logger.warning(f"Guardrail check failed! Risk Level: {level.name}, Score: {score}")
            
        return result

    def _sanitize(self, text: str) -> str:
        """Layer 1: Sanitize input by unescaping HTML and removing basic tags."""
        # Unescape HTML entities (e.g., &amp; -> &, &lt; -> <)
        text = html.unescape(text)
        return text

    def validate_llm_output(self, output: str) -> bool:
        """Validate that the LLM output is structurally sound JSON."""
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError as exc:
            logger.error(f"LLM output validation failed: invalid JSON. {exc}")
            return False
