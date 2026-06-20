"""
Deterministic vehicle tracking, keyframe extraction, and avoidability check.
Two-pass design: pass-1 tracks without storing frames (memory-safe);
pass-2 seeks to the 4 keyframe positions.
"""

import math
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

import avoidability
import config
from schemas import AvoidabilityResult, KeyFrame, VehicleFact

# ── helpers ────────────────────────────────────────────────────────────────────

def _resize(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > config.INPUT_W:
        scale = config.INPUT_W / w
        frame = cv2.resize(frame, (config.INPUT_W, int(h * scale)))
    return frame


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def _velocity(hist: deque) -> tuple[float, float]:
    """Mean (vx, vy) from deque of (frame_idx, cx, cy)."""
    pts = list(hist)
    if len(pts) < 2:
        return 0.0, 0.0
    dx = [pts[i][1] - pts[i-1][1] for i in range(1, len(pts))]
    dy = [pts[i][2] - pts[i-1][2] for i in range(1, len(pts))]
    return float(np.mean(dx)), float(np.mean(dy))


def _velocity_near(hist: deque, impact_fi: int, window: int = 8) -> tuple[float, float]:
    pts = [(fi, cx, cy) for fi, cx, cy in hist if abs(fi - impact_fi) <= window]
    pts.sort(key=lambda x: x[0])
    if len(pts) < 2:
        return _velocity(hist)
    dx = [pts[i][1] - pts[i-1][1] for i in range(1, len(pts))]
    dy = [pts[i][2] - pts[i-1][2] for i in range(1, len(pts))]
    return float(np.mean(dx)), float(np.mean(dy))


def _heading_deg(vx: float, vy: float) -> float:
    return math.degrees(math.atan2(-vy, vx)) % 360


def _speed_kph(px_per_frame: float, fps: float, frame_w: int) -> float:
    metres_per_px = 20.0 / max(frame_w, 1)
    return px_per_frame * fps * metres_per_px * 3.6


def _had_safe_stop(speed_with_fi: list[tuple[int, float]], impact_fi: int) -> bool:
    pre = [s for fi, s in speed_with_fi if fi <= impact_fi]
    if len(pre) < 6:
        return False
    peak = max(pre[: len(pre) // 2] or pre)
    if peak < 0.5:
        return True
    tail = float(np.mean(pre[-3:]))
    return (peak - tail) / (peak + 1e-6) > 0.30


def _pairwise_ttc(
    cx_a: float, cy_a: float, vx_a: float, vy_a: float,
    cx_b: float, cy_b: float, vx_b: float, vy_b: float,
    fps:  float,
) -> float:
    dp  = np.array([cx_a - cx_b, cy_a - cy_b])
    dv  = np.array([vx_a - vx_b, vy_a - vy_b])
    dv2 = float(np.dot(dv, dv))
    if dv2 < 1e-6:
        return -1.0
    t_star = float(-np.dot(dp, dv) / dv2)
    if t_star <= 0 or (t_star / fps) > config.HORIZON_SEC * 2:
        return -1.0
    if float(np.linalg.norm(dp + dv * t_star)) > config.COLL_THRESH_PX:
        return -1.0
    return (t_star / fps) * 1000.0


# ── main entry ─────────────────────────────────────────────────────────────────

def track_and_extract(
    video_path: str,
    model: YOLO,
) -> tuple[list[VehicleFact], list[AvoidabilityResult], list[KeyFrame], float]:
    """
    Returns (vehicle_facts, avoidability_results, keyframes, fps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.INPUT_W

    centroid_hist : dict[int, deque]                  = defaultdict(
        lambda: deque(maxlen=config.TRAIL_LEN)
    )
    speed_hist    : dict[int, list[tuple[int, float]]] = defaultdict(list)
    min_dist_by_frame: list[tuple[int, float]]         = []

    scene_overview_fi:  int   | None = None
    scene_overview_jpg: bytes | None = None
    frame_idx = 0

    # ── pass 1: track ─────────────────────────────────────────────────────────
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % config.SAMPLE_N != 0:
            continue

        frame = _resize(raw)
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=config.VEHICLE_CLASSES, verbose=False, imgsz=frame.shape[1],
        )

        tracked: dict[int, tuple[float, float, float, float]] = {}
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                if box.id is None:
                    continue
                tid = int(box.id[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                centroid_hist[tid].append((frame_idx, cx, cy))
                vx, vy = _velocity(centroid_hist[tid])
                speed_hist[tid].append((frame_idx, math.hypot(vx, vy)))
                tracked[tid] = (cx, cy, vx, vy)

        if scene_overview_fi is None and len(tracked) >= 2:
            scene_overview_fi  = frame_idx
            scene_overview_jpg = _encode_jpeg(frame)

        tids = list(tracked.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = tracked[tids[i]], tracked[tids[j]]
                min_dist_by_frame.append(
                    (frame_idx, math.hypot(a[0] - b[0], a[1] - b[1]))
                )

    cap.release()

    # ── impact frame ──────────────────────────────────────────────────────────
    impact_fi = (
        min(min_dist_by_frame, key=lambda x: x[1])[0]
        if min_dist_by_frame
        else max(1, frame_idx // 2)
    )
    if scene_overview_fi is None:
        scene_overview_fi = 1

    # ── vehicle facts ──────────────────────────────────────────────────────────
    reliable_tids = [
        tid for tid, sh in speed_hist.items()
        if len(sh) >= config.MIN_TRACK_FRAMES
    ]

    impact_vels: dict[int, tuple[float, float, float, float]] = {}
    for tid in reliable_tids:
        hist = centroid_hist[tid]
        if not hist:
            continue
        _, cx, cy = list(hist)[-1]
        vx, vy = _velocity_near(hist, impact_fi)
        impact_vels[tid] = (cx, cy, vx, vy)

    W = width if width > 0 else config.INPUT_W
    vehicle_facts: list[VehicleFact] = []
    speed_kph_map: dict[int, float]  = {}

    for tid in reliable_tids:
        if tid not in impact_vels:
            continue
        cx, cy, vx, vy = impact_vels[tid]
        speed_px = math.hypot(vx, vy)
        kph      = _speed_kph(speed_px, fps, W)
        speed_kph_map[tid] = kph

        min_ttc = -1.0
        for other in reliable_tids:
            if other == tid or other not in impact_vels:
                continue
            ocx, ocy, ovx, ovy = impact_vels[other]
            ttc = _pairwise_ttc(cx, cy, vx, vy, ocx, ocy, ovx, ovy, fps)
            if ttc > 0 and (min_ttc < 0 or ttc < min_ttc):
                min_ttc = ttc

        vehicle_facts.append(VehicleFact(
            vehicle_id         = tid,
            speed_px_per_frame = speed_px,
            speed_kph_est      = kph,
            heading_deg        = _heading_deg(vx, vy),
            ttc_ms             = min_ttc,
            had_safe_stop      = _had_safe_stop(speed_hist[tid], impact_fi),
            last_cx            = cx,
            last_cy            = cy,
            frame_idx          = impact_fi,
        ))

    # ── avoidability check ─────────────────────────────────────────────────────
    avoid_results = avoidability.run_all(impact_vels, speed_kph_map, W)

    # ── pass 2: 4 keyframes ────────────────────────────────────────────────────
    gap = int(fps * 1.5)
    kf_targets = [
        ("scene_overview", scene_overview_fi),
        ("pre_impact",     max(impact_fi - gap, 1)),
        ("impact",         impact_fi),
        ("post_impact",    impact_fi + gap),
    ]

    cap2     = cv2.VideoCapture(video_path)
    keyframes: list[KeyFrame] = []
    for ktype, target_fi in kf_targets:
        if ktype == "scene_overview" and scene_overview_jpg:
            keyframes.append(KeyFrame(
                frame_idx     = scene_overview_fi,
                keyframe_type = ktype,
                image_bytes   = scene_overview_jpg,
                ts_ms         = scene_overview_fi / fps * 1000.0,
            ))
            continue
        cap2.set(cv2.CAP_PROP_POS_FRAMES, max(target_fi - 1, 0))
        ok, raw = cap2.read()
        jpg = _encode_jpeg(_resize(raw)) if ok else b""
        keyframes.append(KeyFrame(
            frame_idx     = target_fi,
            keyframe_type = ktype,
            image_bytes   = jpg,
            ts_ms         = target_fi / fps * 1000.0,
        ))
    cap2.release()

    return vehicle_facts, avoid_results, keyframes, fps
