"""
Lightweight MLOps monitoring for Eyewitness.
Tracks per-run stage metrics: latency, LLM cost, token counts,
fallback rate, VLM confidence, vehicle + avoidability counts.
Flushes to Butterbase monitoring_events table asynchronously.

Usage:
    mon = RunMonitor(run_id, model_version)

    with mon.stage("cv") as s:
        facts, keyframes, fps = cv_pipeline.track_and_extract(...)
        s.vehicle_count    = len(facts)
        s.avoidable_count  = sum(1 for a in avoid if a.avoidable)

    with mon.stage("vlm") as s:
        hypothesis = anthropic_client.run_hypothesis(...)
        s.cost_usd       = hypothesis.cost_usd
        s.input_tokens   = hypothesis.input_tokens
        s.output_tokens  = hypothesis.output_tokens
        s.fallback_used  = hypothesis.fallback_used
        s.vlm_confidence = hypothesis.confidence

    mon.flush()   # non-blocking
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageContext:
    stage:          str
    latency_ms:     float   = 0.0
    cost_usd:       float   = 0.0
    input_tokens:   int     = 0
    output_tokens:  int     = 0
    fallback_used:  bool    = False
    vlm_confidence: float   = 0.0
    vehicle_count:  int     = 0
    avoidable_count:int     = 0
    _t0:            float   = field(default=0.0, repr=False)

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.latency_ms = (time.perf_counter() - self._t0) * 1000
        return False   # never suppress exceptions


class RunMonitor:
    def __init__(self, run_id: str, model_version: str) -> None:
        self.run_id        = run_id
        self.model_version = model_version
        self._stages: list[StageContext] = []

    def stage(self, name: str) -> StageContext:
        ctx = StageContext(stage=name)
        self._stages.append(ctx)
        return ctx

    def summary(self) -> dict:
        total_cost    = sum(s.cost_usd for s in self._stages)
        total_tokens  = sum(s.input_tokens + s.output_tokens for s in self._stages)
        total_latency = sum(s.latency_ms for s in self._stages)
        vlm_stage     = next((s for s in self._stages if s.stage == "vlm"), None)
        return {
            "total_latency_ms": round(total_latency, 1),
            "total_cost_usd":   round(total_cost, 6),
            "total_tokens":     total_tokens,
            "vlm_confidence":   vlm_stage.vlm_confidence if vlm_stage else 0.0,
            "fallback_used":    vlm_stage.fallback_used  if vlm_stage else False,
        }

    def flush(self) -> None:
        """Persist all stage events to Butterbase in a daemon thread."""
        threading.Thread(target=self._write, daemon=True).start()

    def _write(self) -> None:
        from butterbase_client import ButterbaseClient
        bb = ButterbaseClient()
        for ctx in self._stages:
            bb.insert_monitoring_event(
                run_id          = self.run_id,
                model_version   = self.model_version,
                stage           = ctx.stage,
                latency_ms      = ctx.latency_ms,
                cost_usd        = ctx.cost_usd,
                input_tokens    = ctx.input_tokens,
                output_tokens   = ctx.output_tokens,
                fallback_used   = ctx.fallback_used,
                vlm_confidence  = ctx.vlm_confidence,
                vehicle_count   = ctx.vehicle_count,
                avoidable_count = ctx.avoidable_count,
            )
