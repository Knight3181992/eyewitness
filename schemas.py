"""Immutable data contracts for the Eyewitness pipeline."""

from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass(frozen=True)
class VehicleFact:
    vehicle_id:         int
    speed_px_per_frame: float
    speed_kph_est:      float
    heading_deg:        float
    ttc_ms:             float   # -1 = none within horizon
    had_safe_stop:      bool
    last_cx:            float
    last_cy:            float
    frame_idx:          int


@dataclass(frozen=True)
class AvoidabilityResult:
    vehicle_id:       int
    speed_kph:        float
    react_dist_m:     float
    stop_dist_m:      float
    total_needed_m:   float
    available_gap_m:  float
    avoidable:        bool

    @property
    def verdict(self) -> str:
        return "✅ AVOIDABLE" if self.avoidable else "⛔ UNAVOIDABLE"


@dataclass(frozen=True)
class KeyFrame:
    frame_idx:     int
    keyframe_type: str    # scene_overview | pre_impact | impact | post_impact
    image_bytes:   bytes
    ts_ms:         float


@dataclass(frozen=True)
class FaultHypothesis:
    fault_vehicle_id:     Optional[int]
    fault_reason:         str
    confidence:           float
    contributing_factors: list[str]
    severity:             str
    raw_json:             dict
    fallback_used:        bool  = False
    cost_usd:             float = 0.0
    input_tokens:         int   = 0
    output_tokens:        int   = 0


@dataclass
class AnalysisResult:
    run_id:        str  = field(default_factory=lambda: str(uuid.uuid4()))
    model_version: str  = "eyewitness-v1"
    clip_filename: str  = ""
    vehicle_facts: list[VehicleFact]       = field(default_factory=list)
    avoidability:  list[AvoidabilityResult] = field(default_factory=list)
    keyframes:     list[KeyFrame]          = field(default_factory=list)
    hypothesis:    Optional[FaultHypothesis] = None
    fps:           float = 30.0
    error:         Optional[str] = None
