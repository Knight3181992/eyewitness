"""
Orchestration: CV facts → avoidability → VLM hypothesis → Butterbase persistence.
Every stage is timed and monitored. Butterbase writes are fire-and-forget.
"""

import os
import threading

from ultralytics import YOLO

import anthropic_client
import config
import cv_pipeline
from butterbase_client import ButterbaseClient
from monitor import RunMonitor
from schemas import AnalysisResult

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(config.YOLO_MODEL)
    return _model


def analyze_clip(video_path: str) -> AnalysisResult:
    """
    Full pipeline: CV + avoidability → VLM → async persist + monitor.
    Returns AnalysisResult; result.error is set (not raised) on stage failure.
    """
    result = AnalysisResult(clip_filename=os.path.basename(video_path))
    mon    = RunMonitor(result.run_id, result.model_version)

    # ── stage 1: CV + avoidability ────────────────────────────────────────────
    with mon.stage("cv") as cv_ctx:
        try:
            facts, avoid, keyframes, fps = cv_pipeline.track_and_extract(
                video_path, _get_model()
            )
            result.vehicle_facts = facts
            result.avoidability  = avoid
            result.keyframes     = keyframes
            result.fps           = fps
            cv_ctx.vehicle_count    = len(facts)
            cv_ctx.avoidable_count  = sum(1 for a in avoid if a.avoidable)
        except Exception as exc:
            result.error = f"CV pipeline: {exc}"
            mon.flush()
            return result

    # ── stage 2: VLM hypothesis ───────────────────────────────────────────────
    with mon.stage("vlm") as vlm_ctx:
        try:
            h = anthropic_client.run_hypothesis(
                result.vehicle_facts,
                result.avoidability,
                result.keyframes,
                result.run_id,
            )
            result.hypothesis      = h
            vlm_ctx.cost_usd       = h.cost_usd
            vlm_ctx.input_tokens   = h.input_tokens
            vlm_ctx.output_tokens  = h.output_tokens
            vlm_ctx.fallback_used  = h.fallback_used
            vlm_ctx.vlm_confidence = h.confidence
        except Exception as exc:
            result.error = f"VLM stage: {exc}"

    # ── stage 3: persist (non-blocking) ──────────────────────────────────────
    threading.Thread(target=_persist, args=(result, mon), daemon=True).start()

    return result


def _persist(result: AnalysisResult, mon: RunMonitor) -> None:
    bb = ButterbaseClient()
    bb.insert_claim(result.run_id, result.model_version, result.clip_filename)
    for fact in result.vehicle_facts:
        bb.insert_fact(result.run_id, result.model_version, fact)
    for kf in result.keyframes:
        bb.insert_frame(result.run_id, result.model_version, kf)
    if result.hypothesis:
        bb.insert_fault_analysis(result.run_id, result.model_version, result.hypothesis)
    mon.flush()
