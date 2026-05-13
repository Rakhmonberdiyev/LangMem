"""
Output processor: Grounding & Hallucination Filter.

Runs a lightweight LLM check to verify the response is grounded
in the retrieved evidence. If claims appear unsupported, it adds
appropriate caveats rather than silently hallucinating.
"""

from config import llm_client

_GROUNDING_SYSTEM = """\
You are a fact-checking assistant.
Your only job: compare the AI response against the provided evidence.
- If a claim in the response is clearly not supported by the evidence,
  append "(unverified)" after that claim.
- If the response is well-grounded or no evidence was needed, return it unchanged.
- Return ONLY the final response text — no commentary, no preamble."""

_GROUNDING_PROMPT = """\
Evidence / context:
{evidence}

AI response to check:
{response}"""


async def ground_and_filter(
    response: str,
    evidence: str,
    model: str,
) -> str:
    """
    Returns the response, possibly with "(unverified)" caveats added.
    If no evidence was gathered, returns the response unchanged.
    """
    if not evidence or evidence.strip() == "No external data required.":
        return response

    result = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _GROUNDING_SYSTEM},
            {
                "role": "user",
                "content": _GROUNDING_PROMPT.format(
                    evidence=evidence[:3000],   # cap to avoid huge prompts
                    response=response,
                ),
            },
        ],
    )
    return result.choices[0].message.content or response
