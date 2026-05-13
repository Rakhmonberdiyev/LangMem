"""Safety guards: prompt injection detection (input) + output sanity check."""

import re

# --- Input safety (rule-based, fast) ---

_INJECTION_PATTERNS = [
    r"ignore (all |previous |prior )?instructions",
    r"forget (everything|all instructions)",
    r"you are now",
    r"new personality",
    r"(act|pretend|behave) as (a |an )?",
    r"system prompt",
    r"jailbreak",
    r"disregard (your |all )?training",
    r"override (your |all )?guidelines",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def check_input(text: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    Fast rule-based check — runs synchronously before any LLM call.
    """
    for pattern in _COMPILED:
        if pattern.search(text):
            return False, f"Potential prompt injection: '{pattern.pattern}'"
    return True, ""


# --- Output safety (light LLM-based) ---

_OUTPUT_BLOCKLIST = [
    "i am now jailbroken",
    "i have no restrictions",
    "my new instructions are",
]


def check_output(text: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    Catches obvious policy violations in the generated response.
    """
    lower = text.lower()
    for phrase in _OUTPUT_BLOCKLIST:
        if phrase in lower:
            return False, f"Unsafe output phrase detected: '{phrase}'"
    return True, ""


def redact_unsafe_output(text: str) -> str:
    """Replace an unsafe response with a safe fallback message."""
    return (
        "I'm sorry, I can't provide that response. "
        "Please ask me something else."
    )
