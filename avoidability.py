"""
Field-of-Safe-Motion avoidability check.
For each vehicle at T_impact, answers: did a physically realizable
escape route exist given the vehicle's pre-impact kinematics?

This is the counterfactual that grounds fault:
  "Vehicle A had 1.1 s — no safe stop existed. Not primary fault."
  "Vehicle B had 3.4 s and 9.1 m — avoidance WAS possible. Primary fault."

Physics:
  reaction_dist  = v × REACTION_S          (distance covered while reacting)
  stop_dist      = v² / (2 × A_MAX)        (minimum physics stop distance)
  total_needed   = reaction_dist + stop_dist
  avoidable      = available_gap > total_needed
"""

from dataclasses import dataclass

# NHTSA standard perception-reaction time
REACTION_S = 1.5
# Hard emergency braking (dry tarmac)
A_MAX_MPS2 = 7.0
# Rough scene width estimate for px→m conversion (same as cv_pipeline)
SCENE_W_M  = 20.0


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


def check_vehicle(
    vehicle_id:     int,
    speed_kph:      float,
    gap_px:         float,
    frame_width_px: int,
) -> AvoidabilityResult:
    """
    Compute avoidability for a single vehicle.

    gap_px         — centroid-to-centroid distance to nearest other vehicle
                     at the impact frame (pixels)
    frame_width_px — used for rough px → m calibration
    """
    v_mps        = speed_kph / 3.6
    m_per_px     = SCENE_W_M / max(frame_width_px, 1)
    gap_m        = gap_px * m_per_px
    react_dist   = v_mps * REACTION_S
    stop_dist    = v_mps ** 2 / (2 * A_MAX_MPS2)
    total_needed = react_dist + stop_dist

    return AvoidabilityResult(
        vehicle_id      = vehicle_id,
        speed_kph       = round(speed_kph, 1),
        react_dist_m    = round(react_dist, 2),
        stop_dist_m     = round(stop_dist, 2),
        total_needed_m  = round(total_needed, 2),
        available_gap_m = round(gap_m, 2),
        avoidable       = gap_m > total_needed,
    )


def run_all(
    impact_vels:    dict[int, tuple[float, float, float, float]],
    speed_kph_map:  dict[int, float],
    frame_width_px: int,
) -> list[AvoidabilityResult]:
    """
    Run avoidability check for every tracked vehicle.
    impact_vels: {track_id: (cx, cy, vx, vy)} at impact frame
    speed_kph_map: {track_id: speed_kph_est}
    """
    results = []
    tids    = list(impact_vels.keys())
    for tid in tids:
        cx_i, cy_i, _, _ = impact_vels[tid]
        # gap = distance to nearest OTHER vehicle centroid at impact
        other_dists = [
            ((cx_i - impact_vels[o][0])**2 + (cy_i - impact_vels[o][1])**2) ** 0.5
            for o in tids if o != tid
        ]
        gap_px = min(other_dists) if other_dists else 0.0
        results.append(
            check_vehicle(
                vehicle_id     = tid,
                speed_kph      = speed_kph_map.get(tid, 0.0),
                gap_px         = gap_px,
                frame_width_px = frame_width_px,
            )
        )
    return results
