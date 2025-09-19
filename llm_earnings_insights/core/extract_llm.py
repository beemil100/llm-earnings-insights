# core/extract_llm.py
import os, re, json, textwrap, time
from typing import Any, Dict, Optional
from core.schemas import EarningsExtract

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()  # 'ollama' | 'openai' | 'anthropic' | 'openrouter'

SYSTEM_PROMPT = "You extract structured financial info from earnings documents and return strict JSON."

# Stronger instruction: NO prose, NO code fences, omit unknowns (don't write null)
SCHEMA_SNIPPET = """
Return ONLY valid JSON (no code fences, no extra text) matching this shape.
If a field is unknown, OMIT it (do not put null).

{
  "ticker": str,
  "filing_type": str,
  "period_end": str,
  "kpis": {
    "revenue": {"value": float, "unit": str, "yoy_pct": float?, "qoq_pct": float?, "source": {"page": int, "snippet": str}?},
    "eps_diluted_gaap": {"value": float, "unit": str, "source": {"page": int, "snippet": str}?},
    "gross_margin_pct": {"value": float, "unit": "%", "source": {"page": int, "snippet": str}?}
  },
  "guidance": {"status": "raise"|"maintain"|"lower"|"none", "text": str?, "drivers": [str]?, "source": {"page": int}?},
  "risks": [str]?,
  "management_tone": {"score": float, "top_phrases": [str]}?,
  "highlights": [str]?,
  "lowlights": [str]?
}
"""

USER_TEMPLATE = """You will read earnings content and extract the fields above.
Prefer numbers that appear in consolidated statements or guidance sections.
Include a 'source' with page # and a short quote when possible.

TEXT (truncated):
{chunks}
"""

def _retry(times=1, delay=1.0):
    def deco(fn):
        def wrapper(*a, **kw):
            last = None
            for i in range(times+1):
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    last = e
                    if i < times:
                        time.sleep(delay * (2**i))
            raise last
        return wrapper
    return deco

# -------- Provider clients --------
def _call_ollama(prompt: str, system: str, model: str = "llama3.1:8b") -> str:
    import ollama
    resp = ollama.chat(model=model, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ])
    return resp["message"]["content"]

def _call_openai(prompt: str, system: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content

def _call_anthropic(prompt: str, system: str, model: str = "claude-3-haiku-20240307") -> str:
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system,
        messages=[{"role":"user","content":prompt}],
        temperature=0,
    )
    return "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")

def _call_openrouter(prompt: str, system: str, model: str = "openai/gpt-4o-mini") -> str:
    import requests
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://local.dev",
        "X-Title": "LLM Earnings Insights",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":prompt}
        ],
        "temperature": 0
    }
    r = requests.post(url, headers=headers, json=data, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

PROVIDER_FN = {
    "ollama": _call_ollama,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "openrouter": _call_openrouter,
}

# -------- JSON scraping + cleaning --------
def _extract_json_text(raw: str) -> str:
    """
    Extract the JSON object from raw LLM text, even if it includes prose or code fences.
    Strategy:
      - If fenced ```...```, take the content inside the FIRST fence.
      - Else, find the first '{' and the last matching '}' and slice.
    """
    s = raw.strip()

    # if it's fenced code ```json ... ```
    fence_match = re.search(r"```(json)?\s*(\{.*?\})\s*```", s, flags=re.S)
    if fence_match:
        return fence_match.group(2)

    # fallback: slice first { ... last }
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]

    # last resort: return as-is (will fail cleanly)
    return s

def _normalize_units(obj: Any) -> Any:
    """
    Normalize common unit strings like 'EUR million'/'USD million' to a consistent pattern.
    We won't change the numeric value here—just clean the unit text.
    """
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if k == "unit" and isinstance(v, str):
                u = v.strip()
                u = u.replace("EUR million", "EUR_m").replace("USD million", "USD_m")
                u = u.replace("EUR billion", "EUR_b").replace("USD billion", "USD_b")
                new[k] = u
            else:
                new[k] = _normalize_units(v)
        return new
    if isinstance(obj, list):
        return [_normalize_units(x) for x in obj]
    return obj

def _drop_nulls(obj: Any) -> Any:
    """Remove None/null values and lists like [null]."""
    if isinstance(obj, dict):
        return {k: _drop_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [ _drop_nulls(x) for x in obj if x is not None ]
    return obj

@_retry(times=1, delay=1.0)
def _ask_model(chunks: str) -> str:
    prompt = textwrap.dedent(SCHEMA_SNIPPET + "\n\n" + USER_TEMPLATE.format(chunks=chunks[:15000]))
    fn = PROVIDER_FN.get(PROVIDER)
    if not fn:
        raise RuntimeError(f"Unknown LLM_PROVIDER '{PROVIDER}'. Use one of {list(PROVIDER_FN)}.")
    return fn(prompt=prompt, system=SYSTEM_PROMPT)

def extract_earnings_info(chunks: str, ticker: str = "UNKNOWN") -> Any:
    """
    Call the selected provider, parse JSON robustly, clean, and validate.
    Returns EarningsExtract or {'error':..., 'raw':...}
    """
    raw = _ask_model(chunks)
    json_text = _extract_json_text(raw)

    # fix bad null casing
    json_text = json_text.replace("NULL", "null")

    try:
        parsed = json.loads(json_text)
    except Exception as e:
        return {"error": f"JSON parse/validation failed: {e}", "raw": raw}

    # Clean up common issues
    parsed = _drop_nulls(parsed)
    parsed = _normalize_units(parsed)

    # Try to validate against our schema
    try:
        return EarningsExtract(**parsed)
    except Exception as e:
        # If schema validation fails, still return cleaned JSON + reason
        return {"error": f"Schema validation failed: {e}", "json": parsed, "raw": raw}
