"""
Append-only Butterbase REST adapter.
Every write includes run_id + model_version.
All methods are safe to call without BUTTERBASE_API_KEY — they log and return False.
"""

import base64
import json
import urllib.request
from typing import Optional

import config
from schemas import FaultHypothesis, KeyFrame, VehicleFact, AvoidabilityResult


class ButterbaseClient:
    """
    Locked endpoints (Option A):
      POST /claims
      POST /facts
      POST /frames
      POST /fault_analyses   (append-only — overrides are new rows)
      POST /monitoring_events
    """

    def __init__(self) -> None:
        self._base = config.BB_API_URL
        self._key  = config.BB_API_KEY

    def _post(self, table: str, payload: dict) -> Optional[dict]:
        if not self._key:
            print(f"[BB] BUTTERBASE_API_KEY not set — skipping {table}")
            return None
        try:
            data = json.dumps(payload).encode()
            req  = urllib.request.Request(
                f"{self._base}/{table}",
                data    = data,
                headers = {
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {self._key}",
                },
                method = "POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            print(f"[BB:{table}] {exc}")
            return None

    # ── claims ─────────────────────────────────────────────────────────────────

    def insert_claim(self, run_id: str, model_version: str, clip_filename: str) -> bool:
        return self._post("claims", {
            "run_id": run_id, "model_version": model_version,
            "clip_filename": clip_filename, "status": "pending",
        }) is not None

    # ── facts ──────────────────────────────────────────────────────────────────

    def insert_fact(self, run_id: str, model_version: str, fact: VehicleFact) -> bool:
        return self._post("facts", {
            "run_id":             run_id,
            "model_version":      model_version,
            "vehicle_id":         fact.vehicle_id,
            "speed_px_per_frame": fact.speed_px_per_frame,
            "speed_kph_est":      fact.speed_kph_est,
            "heading_deg":        fact.heading_deg,
            "ttc_ms":             fact.ttc_ms if fact.ttc_ms >= 0 else None,
            "had_safe_stop":      fact.had_safe_stop,
            "frame_idx":          fact.frame_idx,
        }) is not None

    # ── frames ─────────────────────────────────────────────────────────────────

    def insert_frame(self, run_id: str, model_version: str, kf: KeyFrame) -> bool:
        return self._post("frames", {
            "run_id":        run_id,
            "model_version": model_version,
            "frame_idx":     kf.frame_idx,
            "keyframe_type": kf.keyframe_type,
            "ts_ms":         kf.ts_ms,
            "image_b64":     (
                base64.b64encode(kf.image_bytes).decode()
                if kf.image_bytes else None
            ),
        }) is not None

    # ── fault_analyses ─────────────────────────────────────────────────────────

    def insert_fault_analysis(
        self,
        run_id:          str,
        model_version:   str,
        h:               FaultHypothesis,
        override_reason: Optional[str] = None,
    ) -> bool:
        return self._post("fault_analyses", {
            "run_id":               run_id,
            "model_version":        model_version,
            "fault_vehicle_id":     h.fault_vehicle_id,
            "fault_reason":         h.fault_reason,
            "confidence":           h.confidence,
            "contributing_factors": h.contributing_factors,
            "severity":             h.severity,
            "fallback_used":        h.fallback_used,
            "override_reason":      override_reason,
        }) is not None

    def append_override(
        self,
        run_id:          str,
        model_version:   str,
        original_h:      Optional[FaultHypothesis],
        override_reason: str,
    ) -> bool:
        if original_h:
            return self.insert_fault_analysis(
                run_id, model_version, original_h, override_reason
            )
        from prompts import FALLBACK_HYPOTHESIS
        from schemas import FaultHypothesis as FH
        stub = FH(
            fault_vehicle_id=None, fault_reason="No prior VLM analysis.",
            confidence=0.0, contributing_factors=["human_override_only"],
            severity="unknown", raw_json={}, fallback_used=True,
        )
        return self.insert_fault_analysis(run_id, model_version, stub, override_reason)

    # ── monitoring_events ──────────────────────────────────────────────────────

    def insert_monitoring_event(
        self,
        run_id:          str,
        model_version:   str,
        stage:           str,
        latency_ms:      float = 0.0,
        cost_usd:        float = 0.0,
        input_tokens:    int   = 0,
        output_tokens:   int   = 0,
        fallback_used:   bool  = False,
        vlm_confidence:  float = 0.0,
        vehicle_count:   int   = 0,
        avoidable_count: int   = 0,
    ) -> bool:
        return self._post("monitoring_events", {
            "run_id":          run_id,
            "model_version":   model_version,
            "stage":           stage,
            "latency_ms":      latency_ms,
            "cost_usd":        cost_usd,
            "input_tokens":    input_tokens,
            "output_tokens":   output_tokens,
            "fallback_used":   fallback_used,
            "vlm_confidence":  vlm_confidence,
            "vehicle_count":   vehicle_count,
            "avoidable_count": avoidable_count,
        }) is not None
