"""
Layer 2 — LLM Orchestration & Planning
========================================
Uses Groq API (FREE) with LLaMA 3.3 70B.

Get your FREE Groq API key at: https://console.groq.com
  1. Sign up (Google login works, no credit card)
  2. Click API Keys → Create API Key
  3. Paste key in the Streamlit sidebar  OR  add to .env file

Author : Saumaya Dube | Rama University
"""

import os
import json
from dotenv import load_dotenv

# Load .env ONCE at import — but NEVER override what is already in os.environ
# This means: .env is used as fallback, sidebar input always wins
load_dotenv(override=False)

MODEL = "llama-3.3-70b-versatile"   # Current free Groq model (updated April 2025)


# ── Get API key (always reads live from os.environ) ──────────
def _get_api_key() -> str:
    """
    Read the Groq API key from os.environ.
    Works whether the key was set via:
      - .env file (loaded above)
      - Streamlit sidebar  →  os.environ["GROQ_API_KEY"] = key
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()

    if not key or key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not found.\n"
            "Please paste your key in the sidebar (top-left of the app).\n"
            "Get a FREE key at https://console.groq.com"
        )
    return key


def _get_client():
    """Create and return a Groq client using the current API key."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq not installed. Run:  pip install groq")

    return Groq(api_key=_get_api_key())


# ════════════════════════════════════════════════════════════
# STEP 1 — Plan: decide which tools to run
# ════════════════════════════════════════════════════════════

_PLAN_PROMPT = """
You are a Senior Data Scientist. Given a dataset profile, choose the best
analytical tools to run. Pick 5-8 tools from this list:

1. missing_value_analysis  — ALWAYS include
2. descriptive_stats       — for numeric columns
3. categorical_analysis    — for text/category columns
4. distribution_analysis   — to check normality
5. correlation_analysis    — when 2+ numeric columns exist
6. outlier_detection       — to find anomalies
7. hypothesis_testing      — when groups can be compared
8. feature_importance      — when 3+ numeric columns exist

Reply with ONLY valid JSON — no text outside it:
{
  "tools_to_run": ["tool1", "tool2", ...],
  "reasoning": "one sentence why"
}
"""

_VALID_TOOLS = {
    "missing_value_analysis", "descriptive_stats", "categorical_analysis",
    "distribution_analysis",  "correlation_analysis", "outlier_detection",
    "hypothesis_testing",     "feature_importance",
}

_DEFAULT_TOOLS = [
    "missing_value_analysis", "descriptive_stats", "categorical_analysis",
    "correlation_analysis",   "outlier_detection",
]


def plan_analysis(profile_text: str, user_goal: str = "") -> dict:
    """
    Ask the LLM to decide which tools to run on this dataset.
    Always returns a valid dict — falls back to default tools on any error.
    """
    user_msg = f"Dataset Profile:\n{profile_text}"
    if user_goal and user_goal.strip():
        user_msg += f"\n\nUser Goal: {user_goal.strip()}"

    try:
        client = _get_client()

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _PLAN_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=400,
        )

        raw = resp.choices[0].message.content.strip()

        # Strip markdown code fences if model added them
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        plan = json.loads(raw)
        tools = [t for t in plan.get("tools_to_run", []) if t in _VALID_TOOLS]

        # Always ensure missing_value_analysis runs first
        if "missing_value_analysis" not in tools:
            tools.insert(0, "missing_value_analysis")

        return {
            "tools_to_run": tools or _DEFAULT_TOOLS[:],
            "reasoning":    plan.get("reasoning", "Selected based on dataset profile."),
        }

    except Exception as exc:
        return {
            "tools_to_run": _DEFAULT_TOOLS[:],
            "reasoning":    "Default plan used.",
            "_error":       str(exc),
        }


# ════════════════════════════════════════════════════════════
# STEP 2 — Synthesize: write the business report
# ════════════════════════════════════════════════════════════

_SYNTHESIS_PROMPT = """
You are a Senior Data Scientist at a top consulting firm.
Write a clear, professional, actionable business analysis report.

RULES:
- Every number MUST come from the tool results — never invent statistics
- Be specific and quantitative, not vague
- Never say "causes" — say "is associated with"
- Write for a business audience

Use EXACTLY these section headings (copy with emojis):

## 📊 Executive Summary
## 🔍 Data Quality Assessment
## 📈 Key Statistical Findings
## 🔗 Relationships & Patterns
## ⚠️ Anomalies & Outliers
## 💡 Business Recommendations
## ✅ Conclusion
"""


def synthesize_results(profile_text: str, tool_summaries: list, user_goal: str = "") -> str:
    """
    Ask the LLM to write a full business report from all tool results.
    Returns a markdown string — falls back to tool summaries if LLM fails.
    """
    combined = "\n\n".join(
        f"--- Tool {i+1} ---\n{s}"
        for i, s in enumerate(tool_summaries) if s
    )

    user_msg = f"Dataset Profile:\n{profile_text}\n\nTool Results:\n{combined}"
    if user_goal and user_goal.strip():
        user_msg += f"\n\nUser Goal: {user_goal.strip()}"

    try:
        client = _get_client()

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYNTHESIS_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=2500,
        )

        return resp.choices[0].message.content.strip()

    except Exception as exc:
        # Fallback: assemble report from raw tool summaries
        lines = [
            "## 📊 Executive Summary",
            f"Analysis completed. See detailed findings from each tool below.",
            "",
            "## 📈 Key Statistical Findings",
        ]
        for s in tool_summaries:
            if s:
                lines.append(s)
                lines.append("")
        lines += [
            "## ⚠️ Note",
            f"AI narrative unavailable: {exc}",
        ]
        return "\n".join(lines)
