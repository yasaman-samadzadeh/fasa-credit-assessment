"""Optional LLM polish for the analyst memos.

Design contract (why this is safe to demo):

- The deterministic template memo (src/explain.py) is the source of truth: it
  contains every number and fact, computed from the model. The LLM's only job
  is to rewrite that memo in fluent analyst prose.
- The prompt forbids adding, removing or altering facts and numbers; output is
  validated by checking that the probability and rating survived verbatim. If
  the call fails, the key is missing, or validation fails, we silently fall
  back to the template memo — the deliverable can never be degraded by an
  outage or a hallucination.
- Keys come from `.env` (python-dotenv); never hardcoded. Supports either
  OPENAI_API_KEY or ANTHROPIC_API_KEY, whichever is present.

Enable with `uv run python run.py --llm`. Off by default so the core
deliverable is reproducible offline.
"""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv

_SYSTEM = (
    "You are a senior credit analyst. Rewrite the structured risk memo you are "
    "given as 2-4 sentences of fluent, professional analyst prose. Use ONLY the "
    "facts and numbers provided — do not add, remove, or reinterpret any figure, "
    "rating, or qualitative signal. "
    "CRITICAL — default probability: copy the percentage exactly as it appears in "
    "the memo headline (e.g. '9%', '63%'). Do NOT round, approximate, or rephrase "
    "it — no 'about 9%', 'roughly 63%', '9 percent', or changing digits. "
    "Keep the rating word (Low/Medium/High) exactly as given."
)


def _rewrite_openai(memo: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=220,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": memo}],
    )
    return resp.choices[0].message.content.strip()


def _rewrite_anthropic(memo: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=220,
        temperature=0.2,
        system=_SYSTEM,
        messages=[{"role": "user", "content": memo}],
    )
    return resp.content[0].text.strip()


def _facts_preserved(template: str, rewritten: str) -> bool:
    """The rating word and probability must survive the rewrite verbatim."""
    m = re.match(r"(Low|Medium|High) risk: estimated default probability (\d+%)", template)
    if m is None:
        return False
    rating, prob = m.group(1), m.group(2)
    return rating in rewritten and prob in rewritten


def polish_memos(memos: list[str]) -> tuple[list[str], int]:
    """Rewrite each memo with the configured LLM; fall back per-memo on any issue.

    Returns (memos, n_polished). n_polished==0 means no key was found or every
    call failed - callers can report this without breaking the pipeline.
    """
    load_dotenv()
    if os.getenv("OPENAI_API_KEY"):
        rewrite = lambda m: _rewrite_openai(m, os.getenv("LLM_MODEL", "gpt-4o-mini"))
    elif os.getenv("ANTHROPIC_API_KEY"):
        rewrite = lambda m: _rewrite_anthropic(m, os.getenv("LLM_MODEL", "claude-3-5-haiku-latest"))
    else:
        return memos, 0

    polished, n_ok = [], 0
    for memo in memos:
        try:
            candidate = rewrite(memo)
            if candidate and _facts_preserved(memo, candidate):
                polished.append(candidate)
                n_ok += 1
            else:
                polished.append(memo)
        except Exception:
            polished.append(memo)
    return polished, n_ok
