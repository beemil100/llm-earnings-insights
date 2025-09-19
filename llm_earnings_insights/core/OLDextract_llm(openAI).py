import json
from typing import Any
from openai import OpenAI
from core.schemas import EarningsExtract


def extract_earnings_info(chunks: str, ticker: str = "UNKNOWN") -> Any:
    """
    Ask LLM to parse text into EarningsExtract schema.
    """
    client = OpenAI()

    schema_prompt = """
    You are a financial analyst. Extract structured JSON that matches this schema:
    {
      "ticker": str,
      "filing_type": str,
      "period_end": str,
      "kpis": {
        "revenue": {"value": float, "unit": "USD_b"},
        "eps_diluted_gaap": {"value": float, "unit": "USD"},
        "gross_margin_pct": {"value": float, "unit": "%"}
      },
      "guidance": {"status": "raise|maintain|lower|none", "text": str},
      "risks": [str],
      "management_tone": {"score": float, "top_phrases": [str]},
      "highlights": [str],
      "lowlights": [str]
    }
    Only respond with JSON, no extra text.
    """

    prompt = schema_prompt + f"\n\nTEXT:\n{chunks}\n"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract structured financial info."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
        return EarningsExtract(**parsed)
    except Exception as e:
        return {"error": str(e), "raw": raw}
