"""Hardened VLM prompts and fallback values for fault analysis."""

import json

FAULT_ANALYSIS_SYSTEM = """\
You are an expert traffic-accident analyst with forensic vision capabilities.
You will receive structured CV facts, physics-based avoidability results,
and dashcam keyframes.

RULES — read carefully before generating output:
1. Output ONLY valid JSON. No markdown, no prose, no code fences, no trailing text.
2. All five fields are required. Use JSON null for fault_vehicle_id only when fault is genuinely shared.
3. confidence is a float in [0.0, 1.0]. Ground it in visual evidence quality.
4. fault_reason must be one sentence, past-tense, observable facts only.
5. contributing_factors must be a non-empty JSON array of short factual strings.
6. severity must be exactly one of: minor, moderate, severe, critical.
7. CRITICAL — treat avoidability as ground truth:
   - avoidable=false means the vehicle physically COULD NOT stop in time.
     Do not assign primary fault to this vehicle for the collision itself.
   - avoidable=true means the vehicle HAD a safe escape route and did not take it.
     This is the primary fault signal.
   - If avoidability and visual evidence disagree, note the discrepancy in fault_reason.

Output schema (exactly this structure, nothing else):
{
  "fault_vehicle_id": <integer or null>,
  "fault_reason":     "<one sentence, facts only>",
  "confidence":       <float 0.0-1.0>,
  "contributing_factors": ["<factor>", ...],
  "severity":         "<minor|moderate|severe|critical>"
}"""


FAULT_ANALYSIS_USER_TEMPLATE = """\
CV FACTS (deterministic ground-truth from tracker):
{facts_json}

AVOIDABILITY ANALYSIS (physics — treat as authoritative):
{avoidability_json}

Analyze the attached keyframes in order: scene overview, pre-impact, impact, post-impact.
Determine fault based strictly on the avoidability results and visual evidence.
Output JSON only — no other text."""


def build_user_message(
    vehicle_facts:       list,
    avoidability_results: list | None = None,
) -> str:
    facts_payload = [
        {
            "vehicle_id":    f.vehicle_id,
            "speed_kph_est": round(f.speed_kph_est, 1),
            "heading_deg":   round(f.heading_deg, 1),
            "ttc_ms":        round(f.ttc_ms, 1) if f.ttc_ms >= 0 else None,
            "had_safe_stop": f.had_safe_stop,
        }
        for f in vehicle_facts
    ]

    avoidability_payload = (
        [
            {
                "vehicle_id":       a.vehicle_id,
                "speed_kph":        a.speed_kph,
                "total_needed_m":   a.total_needed_m,
                "available_gap_m":  a.available_gap_m,
                "avoidable":        a.avoidable,
                "verdict":          a.verdict,
            }
            for a in avoidability_results
        ]
        if avoidability_results
        else []
    )

    return FAULT_ANALYSIS_USER_TEMPLATE.format(
        facts_json        = json.dumps(facts_payload,       indent=2),
        avoidability_json = json.dumps(avoidability_payload, indent=2),
    )


FALLBACK_HYPOTHESIS: dict = {
    "fault_vehicle_id":     None,
    "fault_reason":         "Insufficient or unparseable visual evidence; CV facts recorded.",
    "confidence":           0.0,
    "contributing_factors": ["vlm_parse_error"],
    "severity":             "unknown",
}

REQUIRED_FIELDS  = {"fault_vehicle_id", "fault_reason", "confidence",
                    "contributing_factors", "severity"}
VALID_SEVERITIES = {"minor", "moderate", "severe", "critical", "unknown"}
