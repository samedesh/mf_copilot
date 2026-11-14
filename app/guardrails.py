"""Scope/refusal policy, PII redaction, prompt-injection handling."""
from __future__ import annotations
import re

ADVICE_PATTERNS = [
    r"\bshould i (buy|sell|invest|switch|redeem|hold|put)",
    r"\bwhich (fund|scheme)s?\b.*\b(buy|invest|best|better|choose|pick)",
    r"\b(best|top) (mutual )?(fund|scheme)s?\b",
    r"\brecommend",
    r"\bwill\b.*\b(beat|outperform|go up|rise|grow|double|give (me )?(good )?returns?)\b",
    r"\b(predict|forecast|projection|future returns?|expected returns?)\b",
    r"\bgood time to (invest|buy|sell|enter|exit)",
    r"\bis it (safe|worth|a good)\b",
]
INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|earlier|your)",
    r"you are now\b", r"system prompt", r"reveal (your|the) (prompt|instructions)",
    r"act as (an? )?(unrestricted|jailbroken|different)", r"new instructions\s*:",
]
PII_PATTERNS = [
    (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "[PAN]"),
    (r"\b\d{4}\s\d{4}\s\d{4}\b", "[AADHAAR]"),
    (r"(?i)folio\s*(?:no\.?|number)?\s*[:#]?\s*\d{6,}", "[FOLIO]"),
    (r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b", "[PHONE]"),
    (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[EMAIL]"),
]

REFUSAL_TEXT = ("I can only answer factual questions about scheme documents (exit load, expense ratio, "
                "benchmark, riskometer, holdings, and similar). I can't give investment advice, "
                "recommendations or return predictions. Please consult a SEBI-registered advisor.")


def redact_pii(text: str) -> tuple[str, int]:
    n = 0
    for pat, repl in PII_PATTERNS:
        text, k = re.subn(pat, repl, text)
        n += k
    return text, n


def check_scope(question: str) -> tuple[bool, str]:
    q = question.lower()
    for p in INJECTION_PATTERNS:
        if re.search(p, q):
            return True, "prompt_injection_in_question"
    for p in ADVICE_PATTERNS:
        if re.search(p, q):
            return True, "investment_advice_or_prediction"
    return False, ""


def sanitize_context(text: str) -> tuple[str, int]:
    """Remove instruction-like lines from retrieved document text (indirect prompt injection)."""
    kept, removed = [], 0
    for line in text.splitlines():
        if any(re.search(p, line.lower()) for p in INJECTION_PATTERNS):
            kept.append("[removed: suspected prompt injection]")
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept), removed
