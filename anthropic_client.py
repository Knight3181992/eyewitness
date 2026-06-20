"""
VLM hypothesis stage — production-hardened per Senior ML Engineer patterns:
  - Provider abstraction ready (single _call_api function to swap)
  - tenacity retry with exponential backoff on transient API errors
  - Pydantic output validation (replaces ad-hoc field checks)
  - Cost tracking from response.usage (reads rates from config)
  - Facts-only fallback on any parse or validation failure
  - Never raises — the live pipeline always gets a FaultHypothesis back
"""

import base64
import json
import re
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config
import prompts
from schemas import FaultHypothesis, KeyFrame, VehicleFact
from avoidability import AvoidabilityResult

# ── Pydantic output schema ─────────────────────────────────────────────────────

class _FaultOutput(BaseModel):
    fault_vehicle_id:     Optional[int]  = None
    fault_reason:         str
    confidence:           float          = Field(..., ge=0.0, le=1.0)
    contributing_factors: list[str]      = Field(..., min_length=1)
    severity:             Literal["minor", "moderate", "severe", "critical"]

    @field_validator("contributing_factors")
    @classmethod
    def _non_empty_factors(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("contributing_factors must be non-empty")
        return v


# ── Client singleton ───────────────────────────────────────────────────────────

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY or None)
    return _client


# ── Retry-wrapped API call (provider abstraction layer) ────────────────────────

@retry(
    stop   = stop_after_attempt(config.LLM_MAX_ATTEMPTS),
    wait   = wait_exponential(min=config.LLM_WAIT_MIN_S, max=config.LLM_WAIT_MAX_S),
    retry  = retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError)),
    reraise= True,
)
def _call_api(content: list[dict]) -> anthropic.types.Message:
    """Single entry point for the Anthropic API — swap provider here."""
    return _get_client().messages.create(
        model      = config.CLAUDE_MODEL,
        max_tokens = 400,
        system     = prompts.FAULT_ANALYSIS_SYSTEM,
        messages   = [{"role": "user", "content": content}],
    )


# ── Cost tracking ──────────────────────────────────────────────────────────────

def _compute_cost(usage: anthropic.types.Usage) -> float:
    return (
        usage.input_tokens  * config.COST_INPUT_PER_1K_USD  / 1000 +
        usage.output_tokens * config.COST_OUTPUT_PER_1K_USD / 1000
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",          "", text)
    return text.strip()


def _build_image_block(kf: KeyFrame) -> Optional[dict]:
    if not kf.image_bytes:
        return None
    return {
        "type":   "image",
        "source": {
            "type":        "base64",
            "media_type":  "image/jpeg",
            "data":        base64.b64encode(kf.image_bytes).decode(),
        },
    }


# ── Public entry point ─────────────────────────────────────────────────────────

def run_hypothesis(
    vehicle_facts:        list[VehicleFact],
    avoidability_results: list[AvoidabilityResult],
    keyframes:            list[KeyFrame],
    run_id:               str,
) -> FaultHypothesis:
    """
    Call Claude, validate output with Pydantic, track cost.
    Always returns a valid FaultHypothesis — never raises.
    fallback_used=True + cost_usd=0 signals the VLM output was unusable.
    """
    try:
        content: list[dict] = []
        for kf in keyframes:
            block = _build_image_block(kf)
            if block:
                content.append(block)
        content.append({
            "type": "text",
            "text": prompts.build_user_message(vehicle_facts, avoidability_results),
        })

        resp      = _call_api(content)
        raw_text  = resp.content[0].text.strip()
        cleaned   = _strip_fences(raw_text)
        cost      = _compute_cost(resp.usage)

        try:
            raw_dict = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            print(f"[VLM:{run_id[:8]}] JSON parse error: {exc} | raw: {raw_text[:120]}")
            return _fallback()

        try:
            parsed = _FaultOutput.model_validate(raw_dict)
        except Exception as exc:
            print(f"[VLM:{run_id[:8]}] Pydantic validation: {exc}")
            return _fallback()

        return FaultHypothesis(
            fault_vehicle_id     = parsed.fault_vehicle_id,
            fault_reason         = parsed.fault_reason,
            confidence           = parsed.confidence,
            contributing_factors = parsed.contributing_factors,
            severity             = parsed.severity,
            raw_json             = raw_dict,
            fallback_used        = False,
            cost_usd             = round(cost, 6),
            input_tokens         = resp.usage.input_tokens,
            output_tokens        = resp.usage.output_tokens,
        )

    except Exception as exc:
        print(f"[VLM:{run_id[:8]}] Unrecoverable: {exc}")
        return _fallback()


def _fallback() -> FaultHypothesis:
    d = prompts.FALLBACK_HYPOTHESIS
    return FaultHypothesis(
        fault_vehicle_id     = d["fault_vehicle_id"],
        fault_reason         = d["fault_reason"],
        confidence           = d["confidence"],
        contributing_factors = d["contributing_factors"],
        severity             = d["severity"],
        raw_json             = d,
        fallback_used        = True,
        cost_usd             = 0.0,
        input_tokens         = 0,
        output_tokens        = 0,
    )
