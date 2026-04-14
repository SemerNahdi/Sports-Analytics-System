"""
Sports Analytics System  v6
======================================
Refactored for data integrity, clean video output, and OpenSim compatibility.

Install:
    pip install ultralytics opencv-python numpy pandas scipy matplotlib
    pip install sports2d pose2sim          # for Sports2D pipeline
    # For OpenSim IK: conda install -c opensim-org opensim
"""

import cv2
import math
import json
import os
import threading
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

try:
    from scipy.signal import find_peaks, butter, filtfilt
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    # NOTE: Do NOT call matplotlib.use("Agg") here globally.
    # Sports2D needs an interactive backend to show its native graphs.
    # We set the backend lazily only when our own plotter saves files.
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── YOLO pose detection ────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except (ImportError, Exception):
    HAS_YOLO = False

# ── Optional Sports2D / Pose2Sim ───────────────────────────────────────────────
HAS_SPORTS2D = False
_s2d_angle   = None
_s2d_seg     = None
_SPORTS2D_PROCESS = None

try:
    # Sports2D's Python API import path varies by distribution/version.
    # We support multiple import styles and normalize to a single callable.
    try:
        from Sports2D import Sports2D as _Sports2DModule  # common in some installs
        _SPORTS2D_PROCESS = getattr(_Sports2DModule, "process", None)
    except Exception:
        _Sports2DModule = None

    if _SPORTS2D_PROCESS is None:
        try:
            # Alternative: some versions expose a top-level module API.
            import sports2d as _sports2d_mod  # type: ignore
            _SPORTS2D_PROCESS = getattr(_sports2d_mod, "process", None)
        except Exception:
            _sports2d_mod = None

    HAS_SPORTS2D = _SPORTS2D_PROCESS is not None
    try:
        from Pose2Sim.common import points_to_angles as _pta
        def _s2d_angle(p1, p2, p3):
            v = _pta([p1, p2, p3])
            return [float(v)] if isinstance(v, (float, int, np.number)) else list(v)
        def _s2d_seg(p_from, p_to):
            v = _pta([p_from, p_to])
            return [float(v)] if isinstance(v, (float, int, np.number)) else list(v)
    except Exception:
        pass
except ImportError:
    pass


def _s2d_joint_angle(p_prox, p_vertex, p_dist) -> float:
    if _s2d_angle is not None:
        try:
            pp  = np.array(p_prox,   dtype=float).reshape(1, 2)
            pv  = np.array(p_vertex, dtype=float).reshape(1, 2)
            pd_ = np.array(p_dist,   dtype=float).reshape(1, 2)
            return float(_s2d_angle(pp, pv, pd_)[0])
        except Exception:
            pass
    return angle_3pts(p_prox, p_vertex, p_dist)


def _s2d_seg_angle(p_from, p_to) -> float:
    if _s2d_seg is not None:
        try:
            pf = np.array(p_from, dtype=float).reshape(1, 2)
            pt = np.array(p_to,   dtype=float).reshape(1, 2)
            return float(_s2d_seg(pf, pt)[0])
        except Exception:
            pass
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return float(math.degrees(math.atan2(dx, abs(dy) + 1e-9)))


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

JOINT_NAMES = [
    "head", "neck", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
    "hip_center", "shoulder_center",
]

# COCO keypoint indices for YOLO pose model
_COCO = {
    "nose": 0, "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12, "left_knee": 13,
    "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}


@dataclass
class PoseKeypoints:
    head:            Tuple[float, float] = (0., 0.)
    neck:            Tuple[float, float] = (0., 0.)
    left_shoulder:   Tuple[float, float] = (0., 0.)
    right_shoulder:  Tuple[float, float] = (0., 0.)
    left_elbow:      Tuple[float, float] = (0., 0.)
    right_elbow:     Tuple[float, float] = (0., 0.)
    left_wrist:      Tuple[float, float] = (0., 0.)
    right_wrist:     Tuple[float, float] = (0., 0.)
    left_hip:        Tuple[float, float] = (0., 0.)
    right_hip:       Tuple[float, float] = (0., 0.)
    left_knee:       Tuple[float, float] = (0., 0.)
    right_knee:      Tuple[float, float] = (0., 0.)
    left_ankle:      Tuple[float, float] = (0., 0.)
    right_ankle:     Tuple[float, float] = (0., 0.)
    left_foot:       Tuple[float, float] = (0., 0.)
    right_foot:      Tuple[float, float] = (0., 0.)
    hip_center:      Tuple[float, float] = (0., 0.)
    shoulder_center: Tuple[float, float] = (0., 0.)


@dataclass
class PoseFrame:
    frame_idx: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    kp: PoseKeypoints


@dataclass
class FrameMetrics:
    frame_idx: int = 0
    timestamp: float = 0.
    speed: float = 0.
    acceleration: float = 0.
    stride_length: float = 0.
    step_time: float = 0.
    cadence: float = 0.
    flight_time: float = 0.
    left_knee_angle: float = 0.
    right_knee_angle: float = 0.
    left_hip_angle: float = 0.
    right_hip_angle: float = 0.
    trunk_lean: float = 0.
    direction_change: bool = False
    energy_expenditure: float = 0.
    gait_symmetry: float = 100.
    stride_variability: float = 0.
    fall_risk: float = 0.
    injury_risk: float = 0.
    joint_stress: float = 0.
    fatigue_index: float = 0.
    body_center_disp: float = 0.
    l_valgus: float = 0.
    r_valgus: float = 0.
    risk_score: float = 0.
    l_valgus_clinical: float = 0.
    r_valgus_clinical: float = 0.
    perspective_confidence: float = 1.0


@dataclass
class PlayerSummary:
    player_id: int = 1
    total_frames: int = 0
    duration_seconds: float = 0.
    avg_speed: float = 0.
    max_speed: float = 0.
    avg_stride_length: float = 0.
    avg_step_time: float = 0.
    avg_cadence: float = 0.
    avg_flight_time: float = 0.
    direction_change_freq: float = 0.
    estimated_energy_kcal_hr: float = 0.
    gait_symmetry_pct: float = 0.
    stride_variability_pct: float = 0.
    total_distance_m: float = 0.
    peak_risk_score: float = 0.
    fall_risk_label: str = "Low"
    injury_risk_label: str = "Low"
    injury_risk_detail: str = ""
    body_stress_label: str = "Low"
    fatigue_label: str = "Low"
    double_support_pct: float = 0.
    avg_pelvic_rotation: float = 0.



@dataclass
class BioFrame:
    frame_idx: int = 0
    timestamp: float = 0.
    left_knee_flexion: float = 0.
    right_knee_flexion: float = 0.
    left_hip_flexion: float = 0.
    right_hip_flexion: float = 0.
    left_ankle_dorsiflexion: float = 0.
    right_ankle_dorsiflexion: float = 0.
    left_elbow_flexion: float = 0.
    right_elbow_flexion: float = 0.
    trunk_lateral_lean: float = 0.
    trunk_sagittal_lean: float = 0.
    pelvis_obliquity: float = 0.
    pelvis_rotation: float = 0.
    left_thigh_angle: float = 0.
    right_thigh_angle: float = 0.
    left_shank_angle: float = 0.
    right_shank_angle: float = 0.
    trunk_segment_angle: float = 0.
    left_valgus_clinical: float = 0.
    right_valgus_clinical: float = 0.
    left_arm_swing: float = 0.
    right_arm_swing: float = 0.
    arm_swing_asymmetry: float = 0.
    left_knee_ang_vel: float = 0.
    right_knee_ang_vel: float = 0.
    left_hip_ang_vel: float = 0.
    right_hip_ang_vel: float = 0.
    left_heel_strike: bool = False
    right_heel_strike: bool = False
    left_toe_off: bool = False
    right_toe_off: bool = False
    stance_left: bool = False
    stance_right: bool = False
    double_support: bool = False
    step_width: float = 0.
    foot_progression_angle: float = 0.


# ══════════════════════════════════════════════════════════════════════════════
#  MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def angle_3pts(a, b, c) -> float:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    n = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / n, -1, 1))))


def dist2d(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def smooth_arr(arr, w=5) -> np.ndarray:
    a = np.array(arr, dtype=float)
    if HAS_SCIPY:
        return uniform_filter1d(a, size=w)
    return np.convolve(a, np.ones(w) / w, mode='same')


def clamp01(x) -> float:
    return float(np.clip(x, 0., 1.))


def lerp_color(c1, c2, t):
    t = clamp01(t)
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


def risk_color(s):
    t = clamp01(s / 100.)
    if t < 0.5:
        return lerp_color((0, 200, 0), (0, 200, 255), t * 2)
    return lerp_color((0, 200, 255), (0, 0, 230), (t - .5) * 2)


def bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-6)


def bbox_centre(bbox):
    x, y, w, h = bbox
    return (x + w / 2., y + h / 2.)


def crop_hist(frame, bbox):
    bx, by, bw, bh = [int(v) for v in bbox]
    H, W = frame.shape[:2]
    bx, by = max(0, bx), max(0, by)
    bw, bh = min(bw, W - bx), min(bh, H - by)
    if bw < 5 or bh < 5:
        return None
    hsv = cv2.cvtColor(frame[by:by + bh, bx:bx + bw], cv2.COLOR_BGR2HSV)
    # Slightly higher binning improves discriminative power for kit colors
    # without making the histogram too sparse/noisy.
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def hist_sim(h1, h2) -> float:
    if h1 is None or h2 is None:
        return 0.
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def estimate_player_orientation(kp: PoseKeypoints) -> float:
    sh_w = abs(kp.left_shoulder[0] - kp.right_shoulder[0])
    hp_w = abs(kp.left_hip[0] - kp.right_hip[0])
    body_h = abs(kp.head[1] - kp.left_ankle[1]) + 1e-6
    expected_sh = 0.22 * body_h
    expected_hp = 0.18 * body_h
    conf_sh = clamp01(sh_w / (expected_sh + 1e-6))
    conf_hp = clamp01(hp_w / (expected_hp + 1e-6))
    return float((conf_sh + conf_hp) / 2.0)


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN TRACK
# ══════════════════════════════════════════════════════════════════════════════

class KalmanTrack:
    _next_id = 1
    F = np.array([
        [1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
        [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1],
    ], dtype=float)
    H = np.eye(4, 8)

    def __init__(self, bbox, frame, conf=1.0):
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        cx, cy = bbox_centre(bbox)
        w, h = bbox[2], bbox[3]
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=float)
        self.P = np.diag([10., 10., 10., 10., 100., 100., 10., 10.])
        self.Q = np.diag([1., 1., 1., 1., .5, .5, .2, .2])
        self.R = np.diag([4., 4., 10., 10.])
        self.conf = conf
        self.hit_streak = 1
        self.missed = 0
        self.age = 1
        self.ref_hist = crop_hist(frame, bbox)
        self.last_bbox = bbox
        self.trajectory = deque(maxlen=30)
        self.trajectory.append(bbox_centre(bbox))
        self._yolo_kp = None

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(1., self.x[2])
        self.x[3] = max(1., self.x[3])
        self.age += 1
        self.missed += 1
        return self.get_bbox()

    def update(self, bbox, frame, conf=1.0):
        cx, cy = bbox_centre(bbox)
        z = np.array([cx, cy, bbox[2], bbox[3]], dtype=float)
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T  # numerically stable vs inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.conf = conf
        self.hit_streak += 1
        self.missed = 0
        self.last_bbox = bbox
        self.trajectory.append(bbox_centre(bbox))
        nh = crop_hist(frame, bbox)
        if nh is not None and self.ref_hist is not None:
            self.ref_hist = (0.92 * self.ref_hist + 0.08 * nh).astype(np.float32)
            cv2.normalize(self.ref_hist, self.ref_hist)
        elif nh is not None:
            self.ref_hist = nh

    def get_bbox(self) -> Tuple[int, int, int, int]:
        cx, cy, w, h = self.x[:4]
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))

    def reactivate(self, bbox, frame):
        cx, cy = bbox_centre(bbox)
        self.x[:4] = [cx, cy, bbox[2], bbox[3]]
        # Reset covariance on reactivation to avoid over-trusting stale state
        # after long occlusions (reduces jitter/lag on reacquire).
        self.P = np.diag([10., 10., 10., 10., 100., 100., 10., 10.])
        self.x[4:] = 0.0
        self.missed = 0
        self.hit_streak = 1
        self.last_bbox = bbox
        nh = crop_hist(frame, bbox)
        if nh is not None:
            self.ref_hist = nh


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION LAYER
# ══════════════════════════════════════════════════════════════════════════════

class DetectionLayer:
    def __init__(self, model_size="m"):
        self._yolo = None
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=40, detectShadows=False
        )
        self._mode = "blob"
        if HAS_YOLO:
            try:
                # Look for models in root/models/ or local models/
                # We try several paths to be robust
                mn = f"yolo11{model_size}-pose.pt"
                potential_paths = [
                    mn,
                    os.path.join("models", mn),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", mn), # ../../../models/ (if in src/analytics)
                    os.path.join(os.path.dirname(__file__), "..", "..", "models", mn) # ../../models/
                ]
                
                model_path = mn
                for p in potential_paths:
                    if os.path.exists(p):
                        model_path = p
                        break
                
                self._yolo = _YOLO(model_path)
                self._mode = "yolo"
            except Exception:
                pass

    def reset_bg(self):
        """Reset background model (useful on scene cuts)."""
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=40, detectShadows=False
        )

    @property
    def mode(self):
        return self._mode

    def detect(self, frame) -> List[dict]:
        return self._yolo_detect(frame) if self._mode == "yolo" else self._blob_detect(frame)

    def _yolo_detect(self, frame) -> List[dict]:
        res = self._yolo(frame, verbose=False, conf=0.25)[0]
        dets = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bw, bh = x2 - x1, y2 - y1
            if bh < bw * 0.8:
                continue
            bbox = (int(x1), int(y1), int(bw), int(bh))
            conf = float(box.conf[0].cpu())
            kp = None
            if res.keypoints is not None and i < len(res.keypoints.xy):
                kpxy = res.keypoints.xy[i].cpu().numpy()
                kpc = res.keypoints.conf[i].cpu().numpy()
                kpxy[kpc < 0.3] = 0.
                kp = kpxy
            dets.append({'bbox': bbox, 'conf': conf, 'kp': kp})
        return dets

    def _blob_detect(self, frame) -> List[dict]:
        mask = self._bg.apply(frame)
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (2000 <= area <= 90000):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if not (1.3 <= bh / (bw + 1e-6) <= 5.0):
                continue
            fill = area / (bw * bh + 1e-6)
            if fill < 0.25:
                continue
            cands.append({'bbox': (bx, by, bw, bh), 'conf': fill, 'kp': None, 'area': area})
        cands.sort(key=lambda c: c['area'], reverse=True)
        kept, sup = [], set()
        for i, ci in enumerate(cands):
            if i in sup:
                continue
            kept.append(ci)
            for j, cj in enumerate(cands):
                if j <= i or j in sup:
                    continue
                if bbox_iou(ci['bbox'], cj['bbox']) > 0.40:
                    sup.add(j)
        return kept


# ══════════════════════════════════════════════════════════════════════════════
#  SCENE CHANGE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class SceneChangeDetector:
    def __init__(self, threshold=0.45):
        self._prev = None
        self._thr = threshold

    def is_cut(self, frame) -> bool:
        h, w = frame.shape[:2]
        # Use a centre-crop (80% of frame) to reduce sensitivity to edge banners / overlays
        cy, cx = h // 2, w // 2
        ch, cw = int(h * 0.4), int(w * 0.4)
        crop = frame[cy - ch:cy + ch, cx - cw:cx + cw]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if self._prev is None:
            self._prev = hist
            return False
        score = float(cv2.compareHist(self._prev, hist, cv2.HISTCMP_CORREL))
        self._prev = hist
        return score < self._thr


# ══════════════════════════════════════════════════════════════════════════════
#  BYTETRACKER
# ══════════════════════════════════════════════════════════════════════════════

class ByteTracker:
    HIGH_THRESH = 0.50
    LOW_THRESH  = 0.20
    IOU_HIGH    = 0.30
    IOU_LOW     = 0.15
    IOU_LOST    = 0.20
    MIN_HITS    = 2
    LOST_TTL    = 60

    def __init__(self):
        self.active_tracks: List[KalmanTrack] = []
        self.lost_tracks:   List[KalmanTrack] = []

    def update(self, detections: List[dict], frame) -> List[KalmanTrack]:
        for t in self.active_tracks + self.lost_tracks:
            t.predict()
        high = [d for d in detections if d['conf'] >= self.HIGH_THRESH]
        low  = [d for d in detections if self.LOW_THRESH <= d['conf'] < self.HIGH_THRESH]

        unm_t, unm_h = self._associate(self.active_tracks, high, frame, self.IOU_HIGH)
        still_unm, _ = self._associate(unm_t, low, frame, self.IOU_LOW)
        self._associate(self.lost_tracks, low, frame, self.IOU_LOST, reactivate=True)

        for t in still_unm:
            t.hit_streak = 0
            if t not in self.lost_tracks:
                self.lost_tracks.append(t)
            if t in self.active_tracks:
                self.active_tracks.remove(t)

        for d in unm_h:
            self.active_tracks.append(KalmanTrack(d['bbox'], frame, d['conf']))

        self.lost_tracks = [t for t in self.lost_tracks if t.missed <= self.LOST_TTL]
        return [t for t in self.active_tracks if t.hit_streak >= self.MIN_HITS]

    def _associate(self, tracks, dets, frame, iou_thr, reactivate=False):
        if not tracks or not dets:
            return list(tracks), list(dets)
        
        # We work with a static copy for matching to avoid IndexError if the original 
        # list (e.g. self.lost_tracks) is modified during loop iteration (reactivation).
        tracks_copy = list(tracks)
        cost = np.zeros((len(tracks_copy), len(dets)), dtype=float)
        for ti, t in enumerate(tracks_copy):
            tb = t.get_bbox()
            th = t.ref_hist
            for di, d in enumerate(dets):
                iou = bbox_iou(tb, d['bbox'])
                hs  = hist_sim(th, crop_hist(frame, d['bbox']))
                cost[ti, di] = 1.0 - (iou * 0.60 + hs * 0.40)

        # Use Hungarian matching when scipy is available
        mt, md = set(), set()
        if HAS_SCIPY:
            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost)
                for ti, di in zip(row_ind, col_ind):
                    if cost[ti, di] < 1.0 - iou_thr:
                        mt.add(ti)
                        md.add(di)
                        t = tracks_copy[ti]
                        d = dets[di]
                        if reactivate:
                            t.reactivate(d['bbox'], frame)
                            if t in self.lost_tracks:
                                self.lost_tracks.remove(t)
                            if t not in self.active_tracks:
                                self.active_tracks.append(t)
                        else:
                            t.update(d['bbox'], frame, d['conf'])
                        if d.get('kp') is not None:
                            t._yolo_kp = d['kp']
            except ImportError:
                pass

        if not mt:  # fallback greedy
            while True:
                avail = [(ti, di) for ti in range(len(tracks_copy)) for di in range(len(dets))
                         if ti not in mt and di not in md]
                if not avail:
                    break
                ti, di = min(avail, key=lambda p: cost[p[0], p[1]])
                if cost[ti, di] >= 1.0 - iou_thr:
                    break
                mt.add(ti)
                md.add(di)
                t = tracks_copy[ti]
                d = dets[di]
                if reactivate:
                    t.reactivate(d['bbox'], frame)
                    if t in self.lost_tracks:
                        self.lost_tracks.remove(t)
                    if t not in self.active_tracks:
                        self.active_tracks.append(t)
                else:
                    t.update(d['bbox'], frame, d['conf'])
                if d.get('kp') is not None:
                    t._yolo_kp = d['kp']

        return (
            [tracks[i] for i in range(len(tracks)) if i not in mt],
            [dets[i]   for i in range(len(dets))   if i not in md],
        )

    def reset(self):
        for t in self.active_tracks + self.lost_tracks:
            t.x[4:] = 0.


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET LOCK
# ══════════════════════════════════════════════════════════════════════════════

class TargetLock:
    def __init__(self, seed_bbox, seed_hist, seed_frame_idx, yolo_size="m"):
        self._seed_bbox    = seed_bbox
        self._ref_hist     = seed_hist
        self._seed_fi      = seed_frame_idx
        self._target_id    = None
        self._last_bbox    = None
        self._smooth_box   = None
        self._alpha        = 0.35
        self._state        = "searching"
        self._lost_frames  = 0
        self._fi           = 0
        self.bt    = ByteTracker()
        self.scene = SceneChangeDetector()
        self._det_layer = _get_detection_layer(yolo_size)

    @property
    def state(self):
        return self._state

    @property
    def lost_count(self):
        return self._lost_frames

    def update(self, frame) -> Optional[Tuple]:
        if self.scene.is_cut(frame) and self._fi > 10:
            self.bt.reset()
            self._target_id = None
            self._state = "searching"
            # Scene cut: reset blob background model to avoid drift.
            if self._det_layer is not None:
                try:
                    self._det_layer.reset_bg()
                except Exception:
                    pass

        dets   = self._det_layer.detect(frame)
        tracks = self.bt.update(dets, frame)
        self._fi += 1

        if self._target_id is None:
            if self._fi >= self._seed_fi:
                self._target_id = self._choose(tracks, frame)
                if self._target_id is not None:
                    self._state = "tracking"
            return None

        target = next((t for t in tracks if t.id == self._target_id), None)
        if target is not None:
            target = self._resolve_overlap(target, tracks)

        if target is None:
            self._lost_frames += 1
            self._state = "lost"
            target = self._reacquire(tracks, strict=self._lost_frames <= 5)
            if target is None:
                target = self._reacquire(self.bt.lost_tracks, strict=False)
        else:
            self._lost_frames = 0
            self._state = "tracking"

        if target is None:
            return None

        self._target_id = target.id
        self._last_bbox = target.get_bbox()
        return self._emit(self._last_bbox)

    def _choose(self, tracks, frame) -> Optional[int]:
        if not tracks:
            return None
        best, bid = -1., None
        for t in tracks:
            iou = bbox_iou(t.get_bbox(), self._seed_bbox)
            hs  = hist_sim(t.ref_hist, self._ref_hist)
            sw, sh = self._seed_bbox[2], self._seed_bbox[3]
            tw, th = t.get_bbox()[2], t.get_bbox()[3]
            ss = min(sw * sh, tw * th) / (max(sw * sh, tw * th) + 1e-6)
            sc = iou * 0.45 + hs * 0.40 + ss * 0.15
            if sc > best:
                best, bid = sc, t.id
        return bid

    def _reacquire(self, tracks, strict=True) -> Optional[KalmanTrack]:
        if not tracks:
            return None
        thr = 0.35 if strict else 0.18
        best, bt = -1., None
        for t in tracks:
            hs = hist_sim(t.ref_hist, self._ref_hist)
            if hs < thr:
                continue
            if self._last_bbox is not None:
                lw, lh = self._last_bbox[2], self._last_bbox[3]
                tw, th = t.get_bbox()[2], t.get_bbox()[3]
                ss = min(lw * lh, tw * th) / (max(lw * lh, tw * th) + 1e-6)
                if ss < 0.25:
                    continue
                sc = hs * 0.65 + ss * 0.35
            else:
                sc = hs
            if sc > best:
                best, bt = sc, t
        if bt is not None:
            self._target_id = bt.id
            self._state = "tracking"
        return bt

    def _resolve_overlap(self, target, tracks) -> KalmanTrack:
        tb = target.get_bbox()
        for other in tracks:
            if other.id == target.id:
                continue
            if bbox_iou(tb, other.get_bbox()) > 0.55:
                ts = hist_sim(target.ref_hist, self._ref_hist)
                os = hist_sim(other.ref_hist, self._ref_hist)
                if os > ts + 0.12:
                    self._target_id = other.id
                    return other
        return target

    def _emit(self, bbox) -> Tuple:
        arr = np.array(bbox, dtype=float)
        if self._smooth_box is None:
            self._smooth_box = arr
        else:
            self._smooth_box = self._alpha * arr + (1 - self._alpha) * self._smooth_box
        return tuple(int(v) for v in self._smooth_box)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL DETECTION SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_detection_layers: dict[str, DetectionLayer] = {}


def _get_detection_layer(model_size="m") -> DetectionLayer:
    global _detection_layers
    if model_size not in _detection_layers:
        _detection_layers[model_size] = DetectionLayer(model_size)
    return _detection_layers[model_size]


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PLAYER PICKER
# ══════════════════════════════════════════════════════════════════════════════

def pick_player_interactive(video_path: str) -> Optional[dict]:
    det = _get_detection_layer()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WARMUP = min(90, total // 3)
    cands = []
    for fi in range(WARMUP):
        ret, frame = cap.read()
        if not ret:
            break
        dets = det.detect(frame)
        if dets:
            cands.append((frame.copy(), dets, fi))
    cap.release()
    if not cands:
        return select_primary_player(video_path)

    best_frame, best_dets, best_fi = max(cands, key=lambda c: len(c[1]))
    display = cv2.addWeighted(best_frame.copy(), 0.65, np.zeros_like(best_frame), 0.35, 0)
    COLORS = [
        (0,255,180),(0,140,255),(255,215,0),(0,200,255),
        (180,0,255),(0,255,80),(255,80,80),(80,255,255),
    ]
    blobs = [d['bbox'] for d in best_dets]
    for i, (bx, by, bw, bh) in enumerate(blobs):
        col = COLORS[i % len(COLORS)]
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), col, 3, cv2.LINE_AA)
        lbl = str(i + 1)
        lw, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        bxb, byb = bx + bw // 2 - lw // 2 - 6, max(0, by - 34)
        cv2.rectangle(display, (bxb, byb), (bxb + lw + 12, byb + 28), col, -1)
        cv2.putText(display, lbl, (bxb + 6, byb + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    BH = 52
    banner = np.full((BH, W, 3), 15, np.uint8)
    cv2.putText(banner, "CLICK player to track  |  ESC=auto", (W // 2 - 200, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 215, 0), 1, cv2.LINE_AA)
    display = np.vstack([banner, display])
    chosen = [None]

    def on_click(ev, cx, cy, fl, p):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        ay = cy - BH
        if ay < 0:
            return
        for b in blobs:
            bx, by, bw, bh = b
            if bx <= cx <= bx + bw and by <= ay <= by + bh:
                chosen[0] = b
                return
        chosen[0] = min(blobs, key=lambda b: math.hypot(
            cx - (b[0] + b[2] / 2), ay - (b[1] + b[3] / 2)))

    cv2.namedWindow("Select Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Player", min(W, 1280), min(H + BH, 800))
    cv2.setMouseCallback("Select Player", on_click)
    while True:
        cv2.imshow("Select Player", display)
        if chosen[0] is not None or (cv2.waitKey(20) & 0xFF) == 27:
            break
    cv2.destroyAllWindows()
    if chosen[0] is None:
        return select_primary_player(video_path)
    blob = chosen[0]
    bx, by, bw, bh = blob
    return {'hist': crop_hist(best_frame, blob), 'size': (float(bw), float(bh)),
            'seed_bbox': blob, 'seed_frame': best_fi}


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO PRE-SCAN
# ══════════════════════════════════════════════════════════════════════════════

def select_primary_player(video_path: str, sample_step: int = 6) -> Optional[dict]:
    det = _get_detection_layer()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracks: List[dict] = []
    MAX_GAP = max(sample_step * 5, 30)
    fi = 0
    while fi < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        for d in det.detect(frame):
            blob = d['bbox']
            bx, by, bw, bh = blob
            matched = False
            for tr in tracks:
                if fi - tr["lf"] > MAX_GAP:
                    continue
                iou = bbox_iou(blob, tr["lb"])
                rw, rh = tr["ms"]
                ss = min(bw * bh, rw * rh) / (max(bw * bh, rw * rh) + 1e-6)
                if iou * 0.7 + ss * 0.3 > 0.15 and (iou > 0.10 or ss > 0.55):
                    h = crop_hist(frame, blob)
                    tr["n"] += 1
                    if h is not None:
                        tr["hs"].append(h)
                    n = tr["n"]
                    pw, ph = tr["ms"]
                    tr["ms"] = ((pw * (n - 1) + bw) / n, (ph * (n - 1) + bh) / n)
                    tr["lb"] = blob
                    tr["lf"] = fi
                    matched = True
                    break
            if not matched:
                h = crop_hist(frame, blob)
                tracks.append({
                    "n": 1,
                    "hs": [h] if h is not None else [],
                    "ms": (float(bw), float(bh)),
                    "lb": blob, "lf": fi, "sb": blob, "sf": fi,
                })
        fi += sample_step
    cap.release()
    if not tracks:
        return None
    best = max(tracks, key=lambda t: t["n"])
    mh = None
    if best["hs"]:
        stacked = np.mean(best["hs"], axis=0).astype(np.float32)
        cv2.normalize(stacked, stacked)
        mh = stacked
    return {'hist': mh, 'size': best["ms"], 'seed_bbox': best["sb"], 'seed_frame': best["sf"]}


# ══════════════════════════════════════════════════════════════════════════════
#  HYBRID POSE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class HybridPoseEstimator:
    _VP = dict(head=0.04, neck=0.11, shoulder=0.20, elbow=0.34, wrist=0.46,
               hip=0.54, knee=0.73, ankle=0.91, foot=0.99)

    def __init__(self):
        self._prev_cx = None
        self._dh = deque(maxlen=8)

    def estimate(self, frame, bbox, ts, spd=0., yolo_kp=None) -> PoseKeypoints:
        x, y, w, h = bbox
        cx = x + w / 2.
        disp = abs(cx - self._prev_cx) if self._prev_cx is not None else 0.
        self._prev_cx = cx
        self._dh.append(disp)
        ds = sum(self._dh)
        phase = (ds / max(w * 0.18, 4.)) * math.pi
        swing = clamp01(spd / 9.)
        arm_sw = swing * 0.10 * w
        leg_sw = swing * 0.08 * w
        k_lift = swing * 0.08 * h
        cw = self._cwidths(frame, bbox)
        sh, hh = self._bwidths(cw, w, h)

        def vy(f): return y + f * h

        kp = PoseKeypoints()
        kp.head = (cx, vy(self._VP["head"]))
        kp.neck = (cx, vy(self._VP["neck"]))
        ls = (cx - sh, vy(self._VP["shoulder"]))
        rs = (cx + sh, vy(self._VP["shoulder"]))
        kp.left_shoulder  = ls
        kp.right_shoulder = rs
        kp.shoulder_center = ((ls[0] + rs[0]) / 2., (ls[1] + rs[1]) / 2.)
        aoff = arm_sw * math.sin(phase)
        le = (ls[0] - aoff, vy(self._VP["elbow"]))
        re = (rs[0] + aoff, vy(self._VP["elbow"]))
        kp.left_elbow  = le
        kp.right_elbow = re
        kp.left_wrist  = (le[0] - aoff * .55, vy(self._VP["wrist"]))
        kp.right_wrist = (re[0] + aoff * .55, vy(self._VP["wrist"]))
        lh = (cx - hh, vy(self._VP["hip"]))
        rh = (cx + hh, vy(self._VP["hip"]))
        kp.left_hip   = lh
        kp.right_hip  = rh
        kp.hip_center = ((lh[0] + rh[0]) / 2., (lh[1] + rh[1]) / 2.)
        loff = leg_sw * math.sin(phase)
        roff = -loff
        ll = k_lift * max(0., math.sin(phase))
        rl = k_lift * max(0., -math.sin(phase))
        kp.left_knee   = (lh[0] + loff, vy(self._VP["knee"]) - ll)
        kp.right_knee  = (rh[0] + roff, vy(self._VP["knee"]) - rl)
        kp.left_ankle  = (lh[0] + loff * .45, vy(self._VP["ankle"]) - ll * .5)
        kp.right_ankle = (rh[0] + roff * .45, vy(self._VP["ankle"]) - rl * .5)
        kp.left_foot   = (kp.left_ankle[0] + w * .07,  vy(self._VP["foot"]))
        kp.right_foot  = (kp.right_ankle[0] + w * .07, vy(self._VP["foot"]))

        # Confidence hint: if YOLO keypoints are extremely sparse, downstream
        # angle/risk metrics are likely to be physically meaningless.
        yolo_confident = False
        if yolo_kp is not None and len(yolo_kp) == 17:
            try:
                valid = int(np.sum((yolo_kp[:, 0] > 1) & (yolo_kp[:, 1] > 1)))
                yolo_confident = valid >= 8
            except Exception:
                yolo_confident = False
            def g(nm):
                i = _COCO.get(nm)
                if i is None:
                    return None
                pt = yolo_kp[i]
                return (float(pt[0]), float(pt[1])) if (pt[0] > 1 or pt[1] > 1) else None

            def gxy(nm, df):
                p = g(nm)
                return p if p is not None else df

            kp.left_shoulder  = gxy("left_shoulder",  kp.left_shoulder)
            kp.right_shoulder = gxy("right_shoulder", kp.right_shoulder)
            kp.left_elbow     = gxy("left_elbow",     kp.left_elbow)
            kp.right_elbow    = gxy("right_elbow",    kp.right_elbow)
            kp.left_wrist     = gxy("left_wrist",     kp.left_wrist)
            kp.right_wrist    = gxy("right_wrist",    kp.right_wrist)
            kp.left_hip       = gxy("left_hip",       kp.left_hip)
            kp.right_hip      = gxy("right_hip",      kp.right_hip)
            kp.left_knee      = gxy("left_knee",      kp.left_knee)
            kp.right_knee     = gxy("right_knee",     kp.right_knee)
            kp.left_ankle     = gxy("left_ankle",     kp.left_ankle)
            kp.right_ankle    = gxy("right_ankle",    kp.right_ankle)
            nose = g("nose")
            if nose:
                kp.head = nose
            kp.shoulder_center = (
                (kp.left_shoulder[0] + kp.right_shoulder[0]) / 2.,
                (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2.,
            )
            kp.hip_center = (
                (kp.left_hip[0] + kp.right_hip[0]) / 2.,
                (kp.left_hip[1] + kp.right_hip[1]) / 2.,
            )
            kp.neck = (
                (kp.shoulder_center[0] + kp.head[0]) / 2.,
                (kp.shoulder_center[1] + kp.head[1]) / 2.,
            )
            for side in ("left", "right"):
                ank = getattr(kp, f"{side}_ankle")
                object.__setattr__(kp, f"{side}_foot", (ank[0] + w * .04, ank[1] + h * .03))
        object.__setattr__(kp, "_yolo_confident", bool(yolo_confident))
        return kp

    def _cwidths(self, frame, bbox):
        bx, by, bw, bh = bbox
        H, W = frame.shape[:2]
        bx2, by2 = min(bx + bw, W), min(by + bh, H)
        bx, by = max(0, bx), max(0, by)
        if bx2 - bx < 5 or by2 - by < 5:
            return None
        crop = frame[by:by2, bx:bx2]
        _, mask = cv2.threshold(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        ws = np.array([np.sum(mask[r] > 0) for r in range(mask.shape[0])], dtype=float)
        return smooth_arr(ws, w=max(3, bh // 20)) if len(ws) > 5 else None

    def _bwidths(self, cw, bw, bh):
        dsh = bw * .29
        dh  = bw * .17
        if cw is None or len(cw) < 10:
            return dsh, dh
        n = len(cw)
        u = cw[int(n * .15):int(n * .40)]
        l = cw[int(n * .48):int(n * .68)]
        sh = float(np.max(u)) / 2. if len(u) else dsh
        hh = float(np.max(l)) / 2. if len(l) else dh
        return float(np.clip(sh, bw * .18, bw * .42)), float(np.clip(hh, bw * .10, bw * .32))


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN JOINT SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class JointKalman:
    def __init__(self, pn=1.5, on=8.0):
        self.x = None
        self.v = 0.
        self.P = np.array([[100., 0.], [0., 100.]])
        self.Q = np.diag([pn, pn * 2])
        self.R = on
        self.F = np.array([[1., 1.], [0., 1.]])
        self.H = np.array([[1., 0.]])

    def update(self, z):
        if self.x is None:
            self.x = z
            return z
        st = self.F @ np.array([self.x, self.v])
        Pp = self.F @ self.P @ self.F.T + self.Q
        y  = z - (self.H @ st)[0]
        S  = (self.H @ Pp @ self.H.T)[0, 0] + self.R
        K  = Pp @ self.H.T / S
        st = st + (K * y).flatten()
        self.P = (np.eye(2) - np.outer(K.flatten(), self.H)) @ Pp
        self.x, self.v = float(st[0]), float(st[1])
        return self.x


class PoseKalmanSmoother:
    def __init__(self):
        self._kx = {}
        self._ky = {}

    def smooth(self, kp) -> PoseKeypoints:
        out = PoseKeypoints()
        for nm in JOINT_NAMES:
            raw = getattr(kp, nm)
            if nm not in self._kx:
                self._kx[nm] = JointKalman()
                self._ky[nm] = JointKalman()
            setattr(out, nm, (self._kx[nm].update(raw[0]), self._ky[nm].update(raw[1])))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  SKELETON RENDERER
# ══════════════════════════════════════════════════════════════════════════════

_W  = (240, 240, 240)
_L  = (255, 200, 0)
_R  = (0, 140, 255)
_S  = (180, 240, 180)

BONE_DEFS = [
    ("head",            "neck",            _W, _W, 4),
    ("neck",            "shoulder_center", _W, _S, 4),
    ("shoulder_center", "hip_center",      _S, _S, 5),
    ("left_shoulder",   "left_elbow",      _L, _L, 5),
    ("left_elbow",      "left_wrist",      _L, _L, 4),
    ("right_shoulder",  "right_elbow",     _R, _R, 5),
    ("right_elbow",     "right_wrist",     _R, _R, 4),
    ("left_shoulder",   "right_shoulder",  _L, _R, 4),
    ("left_hip",        "right_hip",       _L, _R, 5),
    ("left_hip",        "left_knee",       _L, _L, 7),
    ("left_knee",       "left_ankle",      _L, _L, 6),
    ("left_ankle",      "left_foot",       _L, _L, 4),
    ("right_hip",       "right_knee",      _R, _R, 7),
    ("right_knee",      "right_ankle",     _R, _R, 6),
    ("right_ankle",     "right_foot",      _R, _R, 4),
]


def draw_gradient_bone(img, p1, p2, c1, c2, th, rt=0.):
    s = max(8, int(dist2d(p1, p2) / 4))
    for i in range(s):
        t  = i / max(s - 1, 1)
        t2 = (i + 1) / max(s - 1, 1)
        col = lerp_color(lerp_color(c1, c2, t), (0, 0, 220), rt * .6)
        cv2.line(img,
                 (int(p1[0] + t  * (p2[0] - p1[0])), int(p1[1] + t  * (p2[1] - p1[1]))),
                 (int(p1[0] + t2 * (p2[0] - p1[0])), int(p1[1] + t2 * (p2[1] - p1[1]))),
                 col, th, cv2.LINE_AA)


def draw_glow_joint(img, pt, r, col, ga=0.45):
    px, py = int(pt[0]), int(pt[1])
    # Glow is drawn on a per-frame overlay to avoid copying the full frame
    # for every joint.
    ov = img  # overlay buffer
    for rr in range(r + 6, r, -2):
        cv2.circle(ov, (px, py), rr, col, -1, cv2.LINE_AA)


def render_skeleton(frame, kp, risk_tint=0.):
    kpd = {n: getattr(kp, n) for n in JOINT_NAMES}
    for a, b, c1, c2, th in BONE_DEFS:
        if a in kpd and b in kpd:
            draw_gradient_bone(frame, kpd[a], kpd[b], c1, c2, th, risk_tint)
    glow = np.zeros_like(frame)
    sz = {
        "head": 4, "neck": 3,
        "left_shoulder": 4, "right_shoulder": 4,
        "left_elbow": 3,    "right_elbow": 3,
        "left_wrist": 3,    "right_wrist": 3,
        "left_hip": 5,      "right_hip": 5,
        "left_knee": 6,     "right_knee": 6,
        "left_ankle": 5,    "right_ankle": 5,
        "left_foot": 3,     "right_foot": 3,
    }
    for nm, r in sz.items():
        if nm in kpd:
            col = lerp_color(
                _L if "left" in nm else _R if "right" in nm else _W,
                (0, 0, 220), risk_tint * .5,
            )
            draw_glow_joint(glow, kpd[nm], r, col)
            px, py = int(kpd[nm][0]), int(kpd[nm][1])
            cv2.circle(frame, (px, py), r,          (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), max(1, r-2), col,            -1, cv2.LINE_AA)
    cv2.addWeighted(frame, 1.0, glow, 0.45 * 0.5, 0, frame)


# ══════════════════════════════════════════════════════════════════════════════
#  BIOMECHANICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class BiomechanicsEngine:
    FILTER_HZ = 6.0

    def __init__(self, fps: float = 25.0, pix_to_m: float = 0.002):
        self.fps       = fps
        self.pix_to_m  = pix_to_m
        self.frames:   List[BioFrame] = []
        self._ah:      dict = {}
        self._la_y:    List[float] = []
        self._ra_y:    List[float] = []
        self._lf_x:    List[float] = []
        self._rf_x:    List[float] = []
        self.lhs:      List[int] = []
        self.rhs:      List[int] = []
        self.lto:      List[int] = []
        self.rto:      List[int] = []

    def process_frame(self, fi: int, ts: float, kp: PoseKeypoints) -> BioFrame:
        bf = BioFrame(frame_idx=fi, timestamp=ts)

        bf.left_knee_flexion         = _s2d_joint_angle(kp.left_hip,  kp.left_knee,  kp.left_ankle)
        bf.right_knee_flexion        = _s2d_joint_angle(kp.right_hip, kp.right_knee, kp.right_ankle)
        bf.left_hip_flexion          = _s2d_joint_angle(kp.shoulder_center, kp.left_hip,  kp.left_knee)
        bf.right_hip_flexion         = _s2d_joint_angle(kp.shoulder_center, kp.right_hip, kp.right_knee)
        bf.left_ankle_dorsiflexion   = _s2d_joint_angle(kp.left_knee,  kp.left_ankle,  kp.left_foot)
        bf.right_ankle_dorsiflexion  = _s2d_joint_angle(kp.right_knee, kp.right_ankle, kp.right_foot)
        bf.left_elbow_flexion        = _s2d_joint_angle(kp.left_shoulder,  kp.left_elbow,  kp.left_wrist)
        bf.right_elbow_flexion       = _s2d_joint_angle(kp.right_shoulder, kp.right_elbow, kp.right_wrist)

        bf.left_thigh_angle    = _s2d_seg_angle(kp.left_hip,   kp.left_knee)
        bf.right_thigh_angle   = _s2d_seg_angle(kp.right_hip,  kp.right_knee)
        bf.left_shank_angle    = _s2d_seg_angle(kp.left_knee,  kp.left_ankle)
        bf.right_shank_angle   = _s2d_seg_angle(kp.right_knee, kp.right_ankle)
        bf.trunk_segment_angle = _s2d_seg_angle(kp.hip_center, kp.shoulder_center)

        dx = kp.shoulder_center[0] - kp.hip_center[0]
        dy = kp.shoulder_center[1] - kp.hip_center[1]
        bf.trunk_lateral_lean  = math.degrees(math.atan2(dx,       abs(dy) + 1e-9))
        bf.trunk_sagittal_lean = math.degrees(math.atan2(abs(dx),  abs(dy) + 1e-9))

        hd = kp.left_hip[1] - kp.right_hip[1]
        hw = dist2d(kp.left_hip, kp.right_hip) + 1e-9
        bf.pelvis_obliquity = math.degrees(math.atan2(abs(hd), hw))
        # Pelvis rotation: estimated from anterior/posterior hip offset (X only)
        hdx = kp.left_hip[0] - kp.right_hip[0]
        bf.pelvis_rotation  = math.degrees(math.atan2(abs(hdx), hw))

        bf.left_valgus_clinical  = self._clinical_valgus(kp.left_hip,  kp.left_knee,  kp.left_ankle)
        bf.right_valgus_clinical = self._clinical_valgus(kp.right_hip, kp.right_knee, kp.right_ankle)

        bf.left_arm_swing      = abs(self._seg_to_vert(kp.left_shoulder,  kp.left_elbow))
        bf.right_arm_swing     = abs(self._seg_to_vert(kp.right_shoulder, kp.right_elbow))
        bf.arm_swing_asymmetry = abs(bf.left_arm_swing - bf.right_arm_swing)

        bf.left_knee_ang_vel   = self._angvel("lk", bf.left_knee_flexion)
        bf.right_knee_ang_vel  = self._angvel("rk", bf.right_knee_flexion)
        bf.left_hip_ang_vel    = self._angvel("lh", bf.left_hip_flexion)
        bf.right_hip_ang_vel   = self._angvel("rh", bf.right_hip_flexion)

        bf.step_width = abs(kp.left_foot[0] - kp.right_foot[0]) * self.pix_to_m

        la = math.degrees(math.atan2(kp.left_foot[0]  - kp.left_ankle[0],
                                     abs(kp.left_foot[1]  - kp.left_ankle[1])  + 1e-9))
        ra = math.degrees(math.atan2(kp.right_foot[0] - kp.right_ankle[0],
                                     abs(kp.right_foot[1] - kp.right_ankle[1]) + 1e-9))
        bf.foot_progression_angle = (abs(la) + abs(ra)) / 2.

        self._la_y.append(kp.left_ankle[1])
        self._ra_y.append(kp.right_ankle[1])
        self._lf_x.append(kp.left_foot[0])
        self._rf_x.append(kp.right_foot[0])
        self.frames.append(bf)
        return bf

    def post_process(self):
        if len(self.frames) < 8:
            return
        for field in [
            "left_knee_flexion", "right_knee_flexion",
            "left_hip_flexion", "right_hip_flexion",
            "left_ankle_dorsiflexion", "right_ankle_dorsiflexion",
            "trunk_lateral_lean", "trunk_sagittal_lean",
            "left_valgus_clinical", "right_valgus_clinical",
        ]:
            raw = np.array([getattr(f, field) for f in self.frames], dtype=float)
            sm  = self._smooth(raw)
            for i, bf in enumerate(self.frames):
                setattr(bf, field, float(sm[i]))

        md = max(4, int(self.fps * 0.18))
        la = np.array(self._la_y)
        ra = np.array(self._ra_y)
        self.lhs = self._peaks( la, md)
        self.rhs = self._peaks( ra, md)
        self.lto = self._peaks(-la, md)
        self.rto = self._peaks(-ra, md)

        lhs_s = set(self.lhs)
        rhs_s = set(self.rhs)
        lto_s = set(self.lto)
        rto_s = set(self.rto)
        sl = self._stance_mask(self.lhs, self.lto, len(self.frames))
        sr = self._stance_mask(self.rhs, self.rto, len(self.frames))

        lf_x = np.array(self._lf_x)
        rf_x = np.array(self._rf_x)
        for i, bf in enumerate(self.frames):
            bf.left_heel_strike  = i in lhs_s
            bf.right_heel_strike = i in rhs_s
            bf.left_toe_off      = i in lto_s
            bf.right_toe_off     = i in rto_s
            bf.stance_left   = sl[i]
            bf.stance_right  = sr[i]
            bf.double_support = sl[i] and sr[i]
            if bf.left_heel_strike or bf.right_heel_strike:
                bf.step_width = abs(lf_x[i] - rf_x[i]) * self.pix_to_m

    def summary_dict(self) -> dict:
        if not self.frames:
            return {}
        skip = {
            "frame_idx", "timestamp",
            "left_heel_strike", "right_heel_strike",
            "left_toe_off", "right_toe_off",
            "stance_left", "stance_right", "double_support",
        }
        out = {}
        for f in BioFrame.__dataclass_fields__:
            if f in skip:
                continue
            v = np.array([getattr(x, f) for x in self.frames], dtype=float)
            out[f"{f}_mean"] = float(np.mean(v))
            out[f"{f}_max"]  = float(np.max(v))
            out[f"{f}_std"]  = float(np.std(v))
        out["lhs_count"]          = len(self.lhs)
        out["rhs_count"]          = len(self.rhs)
        out["double_support_pct"] = 100. * sum(
            1 for x in self.frames if x.double_support
        ) / max(len(self.frames), 1)
        out["valgus_asymmetry"]   = abs(
            out.get("left_valgus_clinical_mean", 0) - out.get("right_valgus_clinical_mean", 0)
        )
        return out

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(f) for f in self.frames])

    @staticmethod
    def _seg_to_vert(p, d) -> float:
        dx = d[0] - p[0]
        dy = d[1] - p[1]
        return float(math.degrees(math.atan2(dx, abs(dy) + 1e-9)))

    @staticmethod
    def _clinical_valgus(hip, knee, ankle) -> float:
        ha  = np.array([ankle[0] - hip[0], ankle[1] - hip[1]], dtype=float)
        hk  = np.array([knee[0]  - hip[0], knee[1]  - hip[1]], dtype=float)
        dev = float(np.cross(ha, hk)) / (np.linalg.norm(ha) + 1e-9)
        return float(math.degrees(math.atan2(dev, np.linalg.norm(hk) + 1e-9)))

    def _angvel(self, key: str, ang: float) -> float:
        prev = self._ah.get(key, ang)
        self._ah[key] = ang
        return (ang - prev) * self.fps

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        if HAS_SCIPY:
            try:
                nyq = self.fps / 2.
                b, a = butter(4, min(self.FILTER_HZ, nyq * .9) / nyq, btype="low")
                return filtfilt(b, a, arr)
            except Exception:
                pass
        w = max(3, int(self.fps * 0.12))
        return smooth_arr(arr, w=w)

    def _peaks(self, sig: np.ndarray, md: int) -> List[int]:
        if HAS_SCIPY:
            try:
                pk, _ = find_peaks(sig, distance=md, prominence=2.)
                return [int(p) for p in pk]
            except Exception:
                pass
        pks = []
        for i in range(1, len(sig) - 1):
            if sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
                if not pks or i - pks[-1] >= md:
                    pks.append(i)
        return pks

    @staticmethod
    def _stance_mask(hs: List[int], to: List[int], n: int) -> List[bool]:
        m = [False] * n
        for h in hs:
            nxt = [t for t in to if t > h]
            end = min(nxt) if nxt else min(h + 20, n - 1)
            for i in range(h, min(end + 1, n)):
                m[i] = True
        return m


# ══════════════════════════════════════════════════════════════════════════════
#  SPORTS2D RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class Sports2DRunner:
    JOINT_ANGLES = [
        "Right ankle", "Left ankle",
        "Right knee",  "Left knee",
        "Right hip",   "Left hip",
        "Right shoulder", "Left shoulder",
        "Right elbow", "Left elbow",
    ]
    SEGMENT_ANGLES = [
        "Right foot",    "Left foot",
        "Right shank",   "Left shank",
        "Right thigh",   "Left thigh",
        "Pelvis", "Trunk", "Shoulders",
        "Right arm",     "Left arm",
        "Right forearm", "Left forearm",
    ]

    def __init__(self, video_path: str, result_dir: str,
                 player_height_m: float = 1.75,
                 participant_mass_kg: float = 75.0,
                 mode: str = "balanced",
                 show_realtime: bool = False,
                 person_ordering: str = "greatest_displacement",
                 do_ik: bool = False,
                 use_augmentation: bool = False,
                 visible_side: str = "auto front"):
        self.video_path          = video_path
        self.result_dir          = result_dir
        self.player_height_m     = player_height_m
        self.participant_mass_kg = participant_mass_kg
        self.mode                = mode
        self.show_realtime       = show_realtime
        self.person_ordering     = person_ordering
        self.do_ik               = do_ik
        self.use_augmentation    = use_augmentation
        self.visible_side        = visible_side
        self.outputs: dict       = {}

    def run(self) -> dict:
        if not HAS_SPORTS2D:
            print("[S2D] Sports2D not installed — skipping.\n"
                  "      Run: pip install sports2d pose2sim")
            return {}

        os.makedirs(self.result_dir, exist_ok=True)

        # Absolute path so Sports2D can locate the video regardless of cwd
        video_abs = str(os.path.abspath(self.video_path))
        result_abs = str(os.path.abspath(self.result_dir))

        # Sports2D CLI accepts `--visible_side auto front` (two tokens). In the
        # Python config this should be a list like ["auto", "front"], not
        # ["auto front"] (which can crash inside Sports2D).
        vs = self.visible_side
        if isinstance(vs, str):
            vs_tokens = [t for t in vs.strip().split() if t]
        else:
            vs_tokens = list(vs)  # type: ignore[arg-type]
        visible_side_cfg = vs_tokens if vs_tokens else ["auto", "front"]

        config = {
            # ── base: I/O, display, and what to save ──────────────────────────
            "base": {
                "video_input":            video_abs,
                "video_dir":              "",          # video_abs already absolute
                "result_dir":             result_abs,
                "nb_persons_to_detect":   1,
                "person_ordering_method": self.person_ordering,
                "first_person_height":    self.player_height_m,
                "visible_side":           visible_side_cfg,
                "load_trc_px":            "",
                "compare":                False,
                "time_range":             [],
                "webcam_id":              0,
                "input_size":             [1280, 720],
                "show_realtime_results":  self.show_realtime,
                "save_vid":               True,
                "save_img":               False,
                "save_pose":              True,
                "save_angles":            True,
            },
            # ── pose: model and detection parameters ──────────────────────────
            "pose": {
                "pose_model":    "Body_with_feet",
                "mode":          self.mode,
                "det_frequency": 4,
                "slowmo_factor": 1,
                "backend":       "auto",
                "device":        "auto",
                "tracking_mode": "sports2d",
                "keypoint_likelihood_threshold": 0.3,
                "average_likelihood_threshold":  0.5,
                "keypoint_number_threshold":     0.3,
            },
            # ── px_to_meters_conversion: separate section (NOT in base) ───────
            "px_to_meters_conversion": {
                "to_meters":         True,
                "make_c3d":          True,
                "save_calib":        True,
                "floor_angle":       "auto",
                "xy_origin":         ["auto"],
                "perspective_value": 10,
                "perspective_unit":  "distance_m",
                "distortions":       [0.0, 0.0, 0.0, 0.0, 0.0],
                "calib_file":        "",
            },
            # ── angles: which angles to compute and display ───────────────────
            "angles": {
                "calculate_angles":   True,          # ← correct location
                "joint_angles":   self.JOINT_ANGLES,
                "segment_angles": self.SEGMENT_ANGLES,
                "correct_segment_angles_with_floor_angle": True,
                "display_angle_values_on": ["body", "list"],
                "fontSize": 0.3,
            },
            # ── post-processing: filtering and graph saving ───────────────────
            # IMPORTANT: show_graphs and save_graphs live HERE, not in base
            "post-processing": {
                "interpolate":             True,
                "interp_gap_smaller_than": 100,
                "fill_large_gaps_with":    "last_value",
                "sections_to_keep":        "all",
                "min_chunk_size":          10,
                "reject_outliers":         True,
                "filter":                  True,
                "show_graphs":             self.show_realtime,  # mirrors realtime flag
                "save_graphs":             True,
                "filter_type":             "butterworth",
                "butterworth": {
                    "cut_off_frequency": 6,
                    "order": 4,
                },
            },
            # ── kinematics: OpenSim IK (requires full OpenSim install) ────────
            "kinematics": {
                "do_ik":               self.do_ik,
                "use_augmentation":    self.use_augmentation,
                "feet_on_floor":       False,
                "use_simple_model":    False,
                "participant_mass":    [self.participant_mass_kg],
                "right_left_symmetry": True,
                "default_height":      self.player_height_m,
                "fastest_frames_to_remove_percent": 0.1,
                "slowest_frames_to_remove_percent": 0.2,
                "large_hip_knee_angles":            45,
                "trimmed_extrema_percent":          0.5,
                "remove_individual_scaling_setup":  True,
                "remove_individual_ik_setup":       True,
            },
            "logging": {
                "use_custom_logging": False,
            },
        }

        try:
            if _SPORTS2D_PROCESS is None:
                raise RuntimeError(
                    "Sports2D Python API not available (could not resolve process()). "
                    "Install/repair with: pip install sports2d pose2sim"
                )
            _SPORTS2D_PROCESS(config)
        except Exception as e:
            import traceback
            print(f"[S2D] Sports2D.process() failed: {e}")
            traceback.print_exc()
            return {}

        self.outputs = self._collect_outputs()
        return self.outputs

    def _collect_outputs(self) -> dict:
        import glob
        rd = self.result_dir
        out = {
            "annotated_video": [],
            "angle_plots_png": [],
            "trc_pose_px":     [],
            "trc_pose_m":      [],
            "mot_angles":      [],
            "calib_toml":      [],
            "c3d":             [],
            "osim_model":      [],
            "osim_mot":        [],
            "osim_setup":      [],
            "all":             [],
        }
        for f in glob.glob(os.path.join(rd, "**", "*"), recursive=True):
            if not os.path.isfile(f):
                continue
            out["all"].append(f)
            fl   = f.lower()
            name = os.path.basename(fl)
            if fl.endswith(".mp4") or fl.endswith(".avi"):
                if "_h264" not in name:
                    out["annotated_video"].append(f)
            elif fl.endswith(".png"):
                out["angle_plots_png"].append(f)
            elif fl.endswith(".trc"):
                if "_px" in name or "pixel" in name:
                    out["trc_pose_px"].append(f)
                else:
                    out["trc_pose_m"].append(f)
            elif fl.endswith(".mot") and "ik" not in name:
                out["mot_angles"].append(f)
            elif fl.endswith(".toml"):
                out["calib_toml"].append(f)
            elif fl.endswith(".c3d"):
                out["c3d"].append(f)
            elif fl.endswith(".osim"):
                out["osim_model"].append(f)
            elif fl.endswith(".mot") and "ik" in name:
                out["osim_mot"].append(f)
            elif fl.endswith(".xml"):
                out["osim_setup"].append(f)
        return out

    def get_seed_from_trc(self) -> Optional[dict]:
        """
        Read the first few frames of Sports2D's pixel-space TRC file and
        return a seed dict compatible with TargetLock / pick_player_interactive.

        The TRC pixel file contains marker X/Y positions in image pixels.
        We use Hip_Center (or the mean of all markers) to locate the player
        in frame 0, build a rough bounding box, and sample a colour histogram
        from that region.
        """
        trcs_px = self.outputs.get("trc_pose_px", [])
        if not trcs_px:
            return None

        path = trcs_px[0]
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            # Find data start (first numeric row after the 5-line header)
            data_start = None
            for i, line in enumerate(lines):
                stripped = line.strip()
                if i >= 4 and stripped and stripped[0].isdigit():
                    data_start = i
                    break
            if data_start is None:
                return None

            # Header row is 2 lines above data
            header_idx = max(0, data_start - 2)
            df = pd.read_csv(path, sep="	", skiprows=header_idx,
                             encoding="utf-8", on_bad_lines="skip")
            df.columns = [c.strip() for c in df.columns]
            df = df.dropna(axis=1, how="all")

            if df.empty or len(df) < 2:
                return None

            # Robust TRC parsing: infer marker base names from "<Marker>.<X|Y|Z>".
            # We avoid relying on numeric column ordering, because missing markers
            # can shift indices and break modulo assumptions.
            cols = [c for c in df.columns if isinstance(c, str)]
            bases = {}
            for c in cols:
                if c.endswith(".X") or c.endswith(".Y") or c.endswith(".Z"):
                    base, axis = c.rsplit(".", 1)
                    bases.setdefault(base, {})[axis] = c

            if not bases:
                return None

            # Use first valid frame row; coerce to numeric where possible.
            row = df.iloc[0]
            def _val(col):
                try:
                    return float(pd.to_numeric(row.get(col), errors="coerce"))
                except Exception:
                    return float("nan")

            preferred = ["Hip_Center", "Hip_Centre", "L_Hip", "R_Hip", "Left_Hip", "Right_Hip"]
            xs: List[float] = []
            ys: List[float] = []

            def _add_marker(base: str):
                ax = bases.get(base, {})
                xcol = ax.get("X")
                ycol = ax.get("Y")
                if not xcol or not ycol:
                    return
                x = _val(xcol)
                y = _val(ycol)
                if not np.isnan(x) and not np.isnan(y) and x > 0 and y > 0:
                    xs.append(x)
                    ys.append(y)

            # Prefer hip-based markers (most stable for bbox seeding).
            for b in preferred:
                _add_marker(b)

            # Fallback: use all markers with valid X/Y.
            if not xs or not ys:
                for b in bases.keys():
                    _add_marker(b)

            if not xs or not ys:
                return None

            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            # Estimate a reasonable bounding box (~person is ~0.4w × 0.7h of frame)
            spread_x = float(np.ptp(xs)) if len(xs) > 1 else 80.
            spread_y = float(np.ptp(ys)) if len(ys) > 1 else 180.
            w = max(60., spread_x * 1.5)
            h = max(120., spread_y * 1.3)
            bx = int(cx - w / 2)
            by = int(cy - h / 2)
            seed_bbox = (bx, by, int(w), int(h))

            # Try to sample a histogram from the source video at frame 0
            hist = None
            try:
                cap = cv2.VideoCapture(self.video_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        hist = crop_hist(frame, seed_bbox)
                cap.release()
            except Exception:
                pass

            return {
                "seed_bbox":  seed_bbox,
                "seed_frame": 0,
                "hist":       hist,
            }
        except Exception as e:
            print(f"[S2D] get_seed_from_trc failed: {e}")
            return None

    def load_mot_angles(self) -> Optional[pd.DataFrame]:
        mots = self.outputs.get("mot_angles", [])
        if not mots:
            return None
        path = mots[0]
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            header_idx = next(
                (i for i, l in enumerate(lines) if l.strip().lower().startswith("time")),
                None,
            )
            if header_idx is None:
                return None
            df = pd.read_csv(path, sep="\t", skiprows=header_idx,
                             encoding="utf-8", on_bad_lines="skip")
            df.columns = [c.strip() for c in df.columns]
            return df.dropna(axis=1, how="all")
        except Exception as e:
            print(f"[S2D] Failed to load MOT file {path}: {e}")
            return None

    def load_trc_pose(self, metres: bool = True) -> Optional[pd.DataFrame]:
        key  = "trc_pose_m" if metres else "trc_pose_px"
        trcs = self.outputs.get(key, [])
        if not trcs:
            return None
        path = trcs[0]
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            data_start = None
            for i, line in enumerate(lines):
                stripped = line.strip()
                if i >= 3 and stripped and (stripped[0].isdigit() or
                        stripped.lower().startswith("frame")):
                    data_start = i
                    break
            if data_start is None:
                data_start = 5
            header_line = data_start - 2
            df = pd.read_csv(path, sep="\t", skiprows=header_line,
                             encoding="utf-8", on_bad_lines="skip")
            df.columns = [c.strip() for c in df.columns]
            return df.dropna(axis=1, how="all")
        except Exception as e:
            print(f"[S2D] Failed to load TRC file {path}: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  TRC / MOT WRITER  — native OpenSim-compatible file generation
# ══════════════════════════════════════════════════════════════════════════════

class OpenSimFileWriter:
    """
    Generates valid OpenSim input files from tracked pose data.

    TRC format:
        Standard OpenSim Marker Trajectory (marker positions in metres, 3-D).
        We set Z=0 for all markers (monocular video → 2-D plane).
        Coordinate system: X = horizontal (right), Y = vertical (up, image Y inverted),
        Z = depth (out of plane, zero). This matches the standard OpenSim convention
        used by Sports2D / Pose2Sim.

    MOT format:
        OpenSim Motion file (tab-separated, header block).
        Stores joint angles in degrees, same convention as Sports2D.
    """

    # Subset of our joint names that map to standard OpenSim marker labels
    OPENSIM_MARKERS = [
        "head", "neck",
        "left_shoulder", "right_shoulder",
        "left_elbow",    "right_elbow",
        "left_wrist",    "right_wrist",
        "left_hip",      "right_hip",
        "left_knee",     "right_knee",
        "left_ankle",    "right_ankle",
        "left_foot",     "right_foot",
        "hip_center",    "shoulder_center",
    ]

    # Canonical OpenSim marker label mapping
    _LABEL_MAP = {
        "head":             "Head",
        "neck":             "Neck",
        "left_shoulder":    "L_Shoulder",
        "right_shoulder":   "R_Shoulder",
        "left_elbow":       "L_Elbow",
        "right_elbow":      "R_Elbow",
        "left_wrist":       "L_Wrist",
        "right_wrist":      "R_Wrist",
        "left_hip":         "L_Hip",
        "right_hip":        "R_Hip",
        "left_knee":        "L_Knee",
        "right_knee":       "R_Knee",
        "left_ankle":       "L_Ankle",
        "right_ankle":      "R_Ankle",
        "left_foot":        "L_Foot",
        "right_foot":       "R_Foot",
        "hip_center":       "Hip_Center",
        "shoulder_center":  "Shoulder_Center",
    }

    def write_trc(self, pose_frames: List[PoseFrame], path: str,
                  fps: float, pix_to_m: float, frame_height_px: int) -> bool:
        """
        Write a .trc file with 3-D marker trajectories.

        Coordinate conversion from image (px) to OpenSim (m):
            X_osim =  x_px * pix_to_m          (right is positive)
            Y_osim =  (H - y_px) * pix_to_m    (Y flipped: up is positive)
            Z_osim =  0.0                       (monocular — no depth)
        """
        n_frames  = len(pose_frames)
        n_markers = len(self.OPENSIM_MARKERS)
        H         = frame_height_px

        if n_frames == 0:
            print("[TRC] No pose frames — skipping TRC export.")
            return False

        try:
            with open(path, "w", newline="\r\n") as f:
                # ── Header ────────────────────────────────────────────────────
                # Line 0: file-type header
                f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(path)}\n")
                # Line 1: field names
                f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\t"
                        "Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
                # Line 2: values
                f.write(f"{fps:.6f}\t{fps:.6f}\t{n_frames}\t{n_markers}\t"
                        f"m\t{fps:.6f}\t1\t{n_frames}\n")
                # Line 3: marker labels — Frame# Time M1 '' '' M2 '' '' ...
                labels_row = "Frame#\tTime"
                for nm in self.OPENSIM_MARKERS:
                    lbl = self._LABEL_MAP[nm]
                    labels_row += f"\t{lbl}\t\t"  # label + 2 empty for Y Z
                f.write(labels_row + "\n")
                # Line 4: X/Y/Z sub-headers
                xyz_row = "\t"
                for _ in self.OPENSIM_MARKERS:
                    xyz_row += "\tX\tY\tZ"
                f.write(xyz_row + "\n")
                # Line 5: blank separator (OpenSim expects this)
                f.write("\n")

                # ── Data rows ─────────────────────────────────────────────────
                for pf in pose_frames:
                    row = f"{pf.frame_idx + 1}\t{pf.timestamp:.6f}"
                    for nm in self.OPENSIM_MARKERS:
                        px, py = getattr(pf.kp, nm)
                        x =  px * pix_to_m
                        y = (H - py) * pix_to_m  # flip Y: image Y↓ → OpenSim Y↑
                        z = 0.0
                        row += f"\t{x:.6f}\t{y:.6f}\t{z:.6f}"
                    f.write(row + "\n")
            print(f"[TRC] Written: {path}  ({n_frames} frames, {n_markers} markers)")
            return True
        except Exception as e:
            print(f"[TRC] Failed to write {path}: {e}")
            return False

    def write_mot(self, bio_frames: List[BioFrame], path: str, fps: float) -> bool:
        """
        Write a .mot (OpenSim Motion) file containing joint angles (degrees).

        The column ordering matches the standard Sports2D MOT output so the
        file can be loaded directly in OpenSim's Motion Visualizer or used
        as input to Inverse Kinematics.
        """
        if not bio_frames:
            print("[MOT] No biomechanics frames — skipping MOT export.")
            return False

        # Columns to export (all continuous angle fields from BioFrame)
        angle_fields = [
            "left_knee_flexion",    "right_knee_flexion",
            "left_hip_flexion",     "right_hip_flexion",
            "left_ankle_dorsiflexion", "right_ankle_dorsiflexion",
            "left_elbow_flexion",   "right_elbow_flexion",
            "trunk_lateral_lean",   "trunk_sagittal_lean",
            "pelvis_obliquity",     "pelvis_rotation",
            "left_thigh_angle",     "right_thigh_angle",
            "left_shank_angle",     "right_shank_angle",
            "trunk_segment_angle",
            "left_valgus_clinical", "right_valgus_clinical",
            "left_arm_swing",       "right_arm_swing",
        ]

        n_rows = len(bio_frames)
        n_cols = 1 + len(angle_fields)  # time + angles

        try:
            with open(path, "w", newline="\r\n") as f:
                # ── OpenSim MOT header ────────────────────────────────────────
                f.write(f"{os.path.basename(path)}\n")
                f.write("version=1\n")
                f.write(f"nRows={n_rows}\n")
                f.write(f"nColumns={n_cols}\n")
                f.write("inDegrees=yes\n")
                f.write("endheader\n")

                # ── Column header row ─────────────────────────────────────────
                header = "time\t" + "\t".join(angle_fields)
                f.write(header + "\n")

                # ── Data rows ─────────────────────────────────────────────────
                for bf in bio_frames:
                    row = f"{bf.timestamp:.6f}"
                    for field in angle_fields:
                        row += f"\t{getattr(bf, field):.6f}"
                    f.write(row + "\n")
            print(f"[MOT] Written: {path}  ({n_rows} rows, {len(angle_fields)} angles)")
            return True
        except Exception as e:
            print(f"[MOT] Failed to write {path}: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS PLOTTER  — saves all plots to /results
# ══════════════════════════════════════════════════════════════════════════════

class AnalyticsPlotter:
    """
    Generates and saves high-resolution analytical plots to a /results directory.
    All plots are saved as 300 DPI PNG and SVG for publication / external use.
    """

    def __init__(self, results_dir: str, player_id: int = 1):
        self.results_dir = results_dir
        self.player_id   = player_id
        os.makedirs(results_dir, exist_ok=True)
        if HAS_MPL:
            import matplotlib
            try:
                matplotlib.use("Agg")
            except Exception:
                pass  # backend already set; acceptable if Sports2D already switched it

    def _save(self, fig, name: str):
        """Save figure as both PNG (300 DPI) and SVG."""
        import matplotlib.pyplot as _plt
        base = os.path.join(self.results_dir, name)
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".svg", bbox_inches="tight")
        _plt.close(fig)
        print(f"[PLOT] Saved → {base}.png / .svg")

    def plot_speed_profile(self, frame_metrics: List[FrameMetrics]):
        if not frame_metrics or not HAS_MPL:
            return
        ts    = [f.timestamp for f in frame_metrics]
        speed = [f.speed     for f in frame_metrics]
        accel = [f.acceleration for f in frame_metrics]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        fig.suptitle(f"Player #{self.player_id} — Speed & Acceleration Profile", fontsize=13)

        ax1.plot(ts, speed, color="#00C8A0", linewidth=1.5, label="Speed (m/s)")
        ax1.fill_between(ts, speed, alpha=0.15, color="#00C8A0")
        ax1.set_ylabel("Speed (m/s)")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        ax2.plot(ts, accel, color="#FF6B35", linewidth=1.2, label="Acceleration (m/s²)")
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Acceleration (m/s²)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, "speed_acceleration_profile")

    def plot_joint_angles(self, frame_metrics: List[FrameMetrics]):
        if not frame_metrics or not HAS_MPL:
            return
        ts  = [f.timestamp        for f in frame_metrics]
        lk  = [f.left_knee_angle  for f in frame_metrics]
        rk  = [f.right_knee_angle for f in frame_metrics]
        lh  = [f.left_hip_angle   for f in frame_metrics]
        rh  = [f.right_hip_angle  for f in frame_metrics]
        trl = [f.trunk_lean       for f in frame_metrics]

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(f"Player #{self.player_id} — Joint Angle Timeseries", fontsize=13)

        axes[0].plot(ts, lk, label="Left Knee",  color="#FFB300", linewidth=1.4)
        axes[0].plot(ts, rk, label="Right Knee", color="#0088FF", linewidth=1.4)
        axes[0].set_ylabel("Knee Flexion (°)")
        axes[0].axhline(120, color="red", linewidth=0.7, linestyle="--", label="Risk threshold 120°")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, lh, label="Left Hip",  color="#FFB300", linewidth=1.4)
        axes[1].plot(ts, rh, label="Right Hip", color="#0088FF", linewidth=1.4)
        axes[1].set_ylabel("Hip Flexion (°)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(ts, trl, label="Trunk Lean", color="#AA44FF", linewidth=1.4)
        axes[2].set_ylabel("Trunk Lean (°)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, "joint_angles_timeseries")

    def plot_biomechanics(self, bio_engine: "BiomechanicsEngine"):
        if not bio_engine or not bio_engine.frames or not HAS_MPL:
            return
        frames = bio_engine.frames
        ts     = [f.timestamp for f in frames]

        # ── Knee flexion ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, [f.left_knee_flexion  for f in frames], label="L Knee Flexion",
                color="#FFB300", linewidth=1.4)
        ax.plot(ts, [f.right_knee_flexion for f in frames], label="R Knee Flexion",
                color="#0088FF", linewidth=1.4)
        ax.set_title(f"Player #{self.player_id} — Knee Flexion (BiomechanicsEngine)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "knee_flexion")

        # ── Valgus (clinical) ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, [f.left_valgus_clinical  for f in frames], label="L Valgus",
                color="#FF6B35", linewidth=1.4)
        ax.plot(ts, [f.right_valgus_clinical for f in frames], label="R Valgus",
                color="#35C2FF", linewidth=1.4)
        ax.axhline( 10, color="red",    linewidth=0.8, linestyle="--", label="±10° risk")
        ax.axhline(-10, color="red",    linewidth=0.8, linestyle="--")
        ax.axhline( 0,  color="gray",   linewidth=0.6, linestyle="-")
        ax.set_title(f"Player #{self.player_id} — Clinical Knee Valgus/Varus")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "clinical_valgus")

        # ── Hip & ankle ───────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(ts, [f.left_hip_flexion  for f in frames], label="L Hip", color="#FFB300", linewidth=1.4)
        axes[0].plot(ts, [f.right_hip_flexion for f in frames], label="R Hip", color="#0088FF", linewidth=1.4)
        axes[0].set_ylabel("Hip Flexion (°)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, [f.left_ankle_dorsiflexion  for f in frames], label="L Ankle", color="#FFB300", linewidth=1.4)
        axes[1].plot(ts, [f.right_ankle_dorsiflexion for f in frames], label="R Ankle", color="#0088FF", linewidth=1.4)
        axes[1].set_ylabel("Ankle Dorsiflexion (°)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Player #{self.player_id} — Hip & Ankle Kinematics")
        plt.tight_layout()
        self._save(fig, "hip_ankle_kinematics")

        # ── Angular velocities ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, [f.left_knee_ang_vel  for f in frames], label="L Knee ω",  color="#FFB300", linewidth=1.2)
        ax.plot(ts, [f.right_knee_ang_vel for f in frames], label="R Knee ω",  color="#0088FF", linewidth=1.2)
        ax.plot(ts, [f.left_hip_ang_vel   for f in frames], label="L Hip ω",   color="#FF8800", linewidth=1.0, linestyle="--")
        ax.plot(ts, [f.right_hip_ang_vel  for f in frames], label="R Hip ω",   color="#0055CC", linewidth=1.0, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_title(f"Player #{self.player_id} — Joint Angular Velocities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Velocity (°/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "angular_velocities")

        # ── Gait events ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.set_title(f"Player #{self.player_id} — Gait Events (Heel Strikes & Toe Offs)")
        lhs_ts = [frames[i].timestamp for i in bio_engine.lhs if i < len(frames)]
        rhs_ts = [frames[i].timestamp for i in bio_engine.rhs if i < len(frames)]
        lto_ts = [frames[i].timestamp for i in bio_engine.lto if i < len(frames)]
        rto_ts = [frames[i].timestamp for i in bio_engine.rto if i < len(frames)]
        for t in lhs_ts:
            ax.axvline(t, color="#FFB300", linewidth=1.2, alpha=0.8, label="L Heel Strike" if t == lhs_ts[0] else "")
        for t in rhs_ts:
            ax.axvline(t, color="#0088FF", linewidth=1.2, alpha=0.8, label="R Heel Strike" if t == rhs_ts[0] else "")
        for t in lto_ts:
            ax.axvline(t, color="#FFB300", linewidth=0.8, linestyle="--", alpha=0.6, label="L Toe Off" if t == lto_ts[0] else "")
        for t in rto_ts:
            ax.axvline(t, color="#0088FF", linewidth=0.8, linestyle="--", alpha=0.6, label="R Toe Off" if t == rto_ts[0] else "")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l and l not in seen:
                seen[l] = True
                unique_handles.append(h)
                unique_labels.append(l)
        ax.legend(unique_handles, unique_labels, loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save(fig, "gait_events")

        # ── Arm swing ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, [f.left_arm_swing  for f in frames], label="L Arm Swing",  color="#FFB300", linewidth=1.4)
        ax.plot(ts, [f.right_arm_swing for f in frames], label="R Arm Swing",  color="#0088FF", linewidth=1.4)
        ax.plot(ts, [f.arm_swing_asymmetry for f in frames], label="Asymmetry", color="red",    linewidth=1.0, linestyle="--")
        ax.set_title(f"Player #{self.player_id} — Arm Swing Excursion")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "arm_swing")

    def plot_risk_scores(self, frame_metrics: List[FrameMetrics]):
        if not frame_metrics or not HAS_MPL:
            return
        ts       = [f.timestamp    for f in frame_metrics]
        risk     = [f.risk_score   for f in frame_metrics]
        inj      = [f.injury_risk  for f in frame_metrics]
        joint_s  = [f.joint_stress for f in frame_metrics]
        fatigue  = [f.fatigue_index for f in frame_metrics]

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        fig.suptitle(f"Player #{self.player_id} — Risk Indicators", fontsize=13)

        axes[0].plot(ts, risk, color="#FF3333", linewidth=1.5, label="Composite Risk Score")
        axes[0].fill_between(ts, risk, alpha=0.12, color="#FF3333")
        axes[0].axhline(50, color="orange", linewidth=0.8, linestyle="--", label="Moderate threshold (50)")
        axes[0].axhline(75, color="red",    linewidth=0.8, linestyle="--", label="High threshold (75)")
        axes[0].set_ylabel("Risk Score (0–100)")
        axes[0].set_ylim(0, 105)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, [v * 100 for v in inj],     label="Acute Injury Risk", color="#FF6B35", linewidth=1.3)
        axes[1].plot(ts, [v * 100 for v in joint_s], label="Joint Stress",      color="#9B59B6", linewidth=1.3)
        axes[1].plot(ts, [v * 100 for v in fatigue], label="Fatigue Index",     color="#2ECC71", linewidth=1.3)
        axes[1].set_ylabel("Sub-scores (%)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, "risk_scores")

    def plot_energy(self, frame_metrics: List[FrameMetrics]):
        if not frame_metrics or not HAS_MPL:
            return
        ts     = [f.timestamp          for f in frame_metrics]
        energy = [f.energy_expenditure for f in frame_metrics]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, energy, color="#F39C12", linewidth=1.5, label="Metabolic Power (W)")
        ax.fill_between(ts, energy, alpha=0.12, color="#F39C12")
        ax.set_title(f"Player #{self.player_id} — Estimated Metabolic Power (Minetti Model)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "metabolic_power")

    def generate_all(self, frame_metrics: List[FrameMetrics],
                     bio_engine: Optional["BiomechanicsEngine"]):
        """Generate and save all standard plots."""
        if not HAS_MPL:
            print("[PLOT] matplotlib not installed — skipping plot generation.")
            print("       Run: pip install matplotlib")
            return
        self.plot_speed_profile(frame_metrics)
        self.plot_joint_angles(frame_metrics)
        self.plot_risk_scores(frame_metrics)
        self.plot_energy(frame_metrics)
        if bio_engine and bio_engine.frames:
            self.plot_biomechanics(bio_engine)
        print(f"[PLOT] All plots saved to: {self.results_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class SportsAnalyzer:
    PIX_TO_M = None

    def __init__(self, video_path: str,
                 output_video_path: str = "output_annotated.mp4",
                 player_id: int = 1,
                 fps_override: Optional[float] = None,
                 pick: bool = False,
                 yolo_size: str = "m",
                 player_height_m: float = 1.75,
                 player_mass_kg: float = 75.0):
        self.video_path         = video_path
        self.output_video_path  = output_video_path
        self.player_id          = player_id
        self.fps_override       = fps_override
        self.player_height_m    = player_height_m
        self.player_mass_kg     = player_mass_kg

        self.pose_est   = HybridPoseEstimator()
        self.smoother   = PoseKalmanSmoother()
        self.pose_frames:    List[PoseFrame]    = []
        self.frame_metrics:  List[FrameMetrics] = []
        self.summary = PlayerSummary(player_id=player_id)

        self._spd_win         = deque(maxlen=30)
        self._risk_win        = deque(maxlen=15)
        self._speed_history   = deque(maxlen=90)
        self._pix_to_m_samples = deque(maxlen=60)
        self._accel_burst     = 0
        self._fps_cache       = 30.

        self.bio_engine:         Optional[BiomechanicsEngine] = None
        self.sports2d_runner:    Optional[Sports2DRunner]     = None
        self._frame_height_px:   int = 0

        self._det_layer = _get_detection_layer(yolo_size)

        print("\n" + "=" * 50)
        print(" SPORTS ANALYTICS: ENGINE READY")
        print("-" * 50)
        print(f" * POSE DETECTION:    {self._det_layer.mode.upper()}")
        print(f" * BIOMECHANICS:      {'SPORTS2D (Clinical-Grade)' if HAS_SPORTS2D else 'NUMPY (Math Fallback)'}")
        print(f" * SIGNAL FILTERING:  {'SCIPY (Advanced Signal)' if HAS_SCIPY else 'NUMPY (Basic Mean)'}")
        print(f" * PLOTTING:          {'MATPLOTLIB' if HAS_MPL else 'NOT AVAILABLE'}")
        print("=" * 50 + "\n")

        if pick:
            primary = pick_player_interactive(video_path)
        else:
            primary = select_primary_player(video_path)
        if primary is None:
            raise RuntimeError("No player candidates found in video.")
        self.lock = TargetLock(primary["seed_bbox"], primary["hist"], primary["seed_frame"], yolo_size=yolo_size)

    # ── Video processing ──────────────────────────────────────────────────────

    def process_video(self, stride: int = 1, target_height: int = 720, cancel_event: Optional[threading.Event] = None) -> PlayerSummary:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(self.video_path)

        fps   = self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.
        self._fps_cache = fps
        W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Adaptive Resizing: speed up processing by resizing frame once at the start.
        # This reduces YOLO latency and drawing time significantly for 4K/1080p sources.
        scale = 1.0
        if H_orig > target_height and target_height > 0:
            scale = target_height / H_orig
        
        W, H = int(W_orig * scale), int(H_orig * scale)
        self._frame_height_px = H

        out = self._create_writer(self.output_video_path, fps / stride, W, H)
        self.bio_engine = BiomechanicsEngine(fps=fps / stride, pix_to_m=self.PIX_TO_M or 0.002)

        idx = 0
        while True:
            # Check for cancellation signal every frame
            if cancel_event and cancel_event.is_set():
                print(f"[ENGINE] Cancellation signal received at frame {idx}. Aborting...")
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            if idx % stride != 0:
                idx += 1
                continue

            # Resize frame for analysis and final video output
            if scale != 1.0:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

            ts   = idx / fps
            bbox = self.lock.update(frame)

            if bbox and bbox[2] > 20 and bbox[3] > 40:
                target  = next((t for t in self.lock.bt.active_tracks
                                if t.id == self.lock._target_id), None)
                yolo_kp = getattr(target, '_yolo_kp', None) if target else None
                spd     = self.frame_metrics[-1].speed if self.frame_metrics else 0.

                raw_kp = self.pose_est.estimate(frame, bbox, ts, spd, yolo_kp=yolo_kp)
                kp     = self.smoother.smooth(raw_kp)
                pf     = PoseFrame(idx, ts, bbox, kp)
                self.pose_frames.append(pf)

                self._calibrate(kp)
                fm = self._metrics(pf, idx, ts, fps)
                self.frame_metrics.append(fm)
                self._speed_history.append(fm.speed)
                self.bio_engine.process_frame(idx, ts, kp)

                if abs(fm.acceleration) > 4.0:
                    self._accel_burst = 8
                elif self._accel_burst > 0:
                    self._accel_burst -= 1

                # Annotate frame: skeleton + labels only (no side panel)
                frame = self._annotate(frame, pf, fm)
                frame = self._draw_player_aura(frame, kp, fm)

            out.write(frame)
            idx += 1

        cap.release()
        out.release()

        # If we were cancelled, don't proceed to post-processing or summary
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Job was cancelled by the user.")

        # Build summary and process gait before potentially re-encoding
        if self.bio_engine:
            self.bio_engine.post_process()
        self._post_gait(fps)
        self._build_summary()

        return self.summary

    def _create_writer(self, path: str, fps: float, W: int, H: int):
        # For Cloudinary compatibility, we prioritize reliability (mp4v) over browser-native codecs (avc1)
        # because Cloudinary will transcode it to a web-safe format automatically.
        for codec in ["mp4v", "avc1", "H264", "VP80", "X264", "DIVX", "MJPG"]:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
                if writer.isOpened():
                    print(f"[VIDEO] Using codec: {codec}")
                    return writer
                writer.release()
            except Exception:
                continue

        class _NullWriter:
            def write(self, _): pass
            def release(self): pass
        return _NullWriter()


    # ── Annotation (skeleton + labels on video frame, NO side panel) ──────────

    def _annotate(self, frame, pf: PoseFrame, fm: FrameMetrics) -> np.ndarray:
        kp = pf.kp
        rt = clamp01(fm.risk_score / 100.)
        render_skeleton(frame, kp, risk_tint=rt)

        # Knee angle labels
        for kpt, ang in [(kp.left_knee, fm.left_knee_angle),
                         (kp.right_knee, fm.right_knee_angle)]:
            kx, ky = int(kpt[0]), int(kpt[1])
            ac = (0, 220, 0) if ang > 145 else (0, 140, 255) if ang > 120 else (0, 0, 220)
            cv2.putText(frame, f"{ang:.0f}°", (kx + 12, ky - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ac, 1, cv2.LINE_AA)

        # Player badge above head
        hx       = int(kp.head[0])  # use head X, not hip, for correct lateral position
        head_y   = int(kp.head[1]) - 35
        badge    = f"  #{self.player_id}  "
        (tw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bx0      = hx - tw // 2
        overlay  = frame.copy()
        cv2.rectangle(overlay, (bx0, head_y - 22), (bx0 + tw, head_y + 6), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (bx0, head_y - 22), (bx0 + tw, head_y + 6), (255, 215, 0), 1)
        cv2.putText(frame, badge, (bx0 + 2, head_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # Speed & risk overlay near player bbox
        bx, by, bw, bh = pf.bbox
        spd_txt = f"{fm.speed:.1f} m/s"
        cv2.putText(frame, spd_txt, (bx, by - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        return frame

    def _draw_player_aura(self, frame, kp: PoseKeypoints, fm: FrameMetrics) -> np.ndarray:
        if fm.speed < .5:
            return frame
        hx, hy = int(kp.hip_center[0]), int(kp.hip_center[1])
        bx, by, bw, bh = self.pose_frames[-1].bbox
        rx  = max(12, bw // 2 + 6)
        ry  = max(20, bh // 2 + 10)
        col = lerp_color((0, 180, 60), (0, 60, 255), clamp01(fm.speed / 8.))
        if self._accel_burst > 0:
            br = int(rx * 1.6 + self._accel_burst * 3)
            ba = self._accel_burst / 8. * .4
            ov = frame.copy()
            cv2.ellipse(ov, (hx, hy), (br, int(br * 1.4)), 0, 0, 360,
                        (0, 200, 255), 3, cv2.LINE_AA)
            frame[:] = cv2.addWeighted(ov, ba, frame, 1 - ba, 0)
        for exp, a in [(14, .12), (6, .20)]:
            ov = frame.copy()
            cv2.ellipse(ov, (hx, hy), (rx + exp, ry + exp), 0, 0, 360, col, -1, cv2.LINE_AA)
            frame[:] = cv2.addWeighted(ov, a, frame, 1 - a, 0)
        return frame

    # ── Calibration ───────────────────────────────────────────────────────────

    def _calibrate(self, kp: PoseKeypoints):
        left_leg  = dist2d(kp.left_hip,  kp.left_ankle)
        right_leg = dist2d(kp.right_hip, kp.right_ankle)
        leg = max(left_leg, right_leg)
        if leg < 10:
            return
        estimate = 0.90 / leg
        self._pix_to_m_samples.append(estimate)
        self.PIX_TO_M = float(np.median(self._pix_to_m_samples))
        if self.bio_engine is not None:
            self.bio_engine.pix_to_m = self.PIX_TO_M

    # ── Per-frame metrics ─────────────────────────────────────────────────────

    def _metrics(self, pf: PoseFrame, idx: int, ts: float, fps: float) -> FrameMetrics:
        fm  = FrameMetrics(frame_idx=idx, timestamp=ts)
        kp  = pf.kp
        sc  = self.PIX_TO_M or 0.002

        # If pose is low-confidence (e.g., sparse YOLO keypoints), avoid producing
        # biologically invalid angles/risks. We keep kinematics/risk fields stable
        # by carrying forward the last valid metrics, while still updating speed.
        if getattr(kp, "_yolo_confident", True) is False and self.frame_metrics:
            prev_fm = self.frame_metrics[-1]
            fm.left_knee_angle = prev_fm.left_knee_angle
            fm.right_knee_angle = prev_fm.right_knee_angle
            fm.left_hip_angle = prev_fm.left_hip_angle
            fm.right_hip_angle = prev_fm.right_hip_angle
            fm.trunk_lean = prev_fm.trunk_lean
            fm.perspective_confidence = prev_fm.perspective_confidence
            fm.l_valgus_clinical = prev_fm.l_valgus_clinical
            fm.r_valgus_clinical = prev_fm.r_valgus_clinical
            fm.l_valgus = prev_fm.l_valgus
            fm.r_valgus = prev_fm.r_valgus
            fm.energy_expenditure = prev_fm.energy_expenditure
            fm.joint_stress = prev_fm.joint_stress
            fm.fatigue_index = prev_fm.fatigue_index
            fm.injury_risk = prev_fm.injury_risk
            fm.risk_score = prev_fm.risk_score
            fm.fall_risk = prev_fm.fall_risk

            if len(self.pose_frames) >= 2:
                prev = self.pose_frames[-2]
                dt   = ts - prev.timestamp + 1e-9
                dp   = dist2d(kp.hip_center, prev.kp.hip_center) * sc
                raw  = dp / dt
                self._spd_win.append(raw)
                fm.speed            = float(np.mean(self._spd_win))
                fm.body_center_disp = dp
                if len(self.pose_frames) >= 3:
                    p2  = self.pose_frames[-3]
                    dp2 = dist2d(prev.kp.hip_center, p2.kp.hip_center) * sc
                    dt2 = prev.timestamp - p2.timestamp + 1e-9
                    fm.acceleration = (raw - dp2 / dt2) / dt
            return fm

        fm.left_knee_angle  = _s2d_joint_angle(kp.left_hip,  kp.left_knee,  kp.left_ankle)
        fm.right_knee_angle = _s2d_joint_angle(kp.right_hip, kp.right_knee, kp.right_ankle)
        fm.left_hip_angle   = _s2d_joint_angle(kp.left_shoulder,  kp.left_hip,  kp.left_knee)
        fm.right_hip_angle  = _s2d_joint_angle(kp.right_shoulder, kp.right_hip, kp.right_knee)

        dx = kp.shoulder_center[0] - kp.hip_center[0]
        dy = kp.shoulder_center[1] - kp.hip_center[1]
        fm.trunk_lean = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))

        fm.perspective_confidence = estimate_player_orientation(kp)

        lvc = BiomechanicsEngine._clinical_valgus(kp.left_hip,  kp.left_knee,  kp.left_ankle)
        rvc = BiomechanicsEngine._clinical_valgus(kp.right_hip, kp.right_knee, kp.right_ankle)
        fm.l_valgus_clinical = lvc * fm.perspective_confidence
        fm.r_valgus_clinical = rvc * fm.perspective_confidence

        hw = dist2d(kp.left_hip, kp.right_hip) + 1e-6
        fm.l_valgus = abs(kp.left_knee[0]  - kp.left_hip[0])  / hw
        fm.r_valgus = abs(kp.right_knee[0] - kp.right_hip[0]) / hw

        if len(self.pose_frames) >= 2:
            prev = self.pose_frames[-2]
            dt   = ts - prev.timestamp + 1e-9
            dp   = dist2d(kp.hip_center, prev.kp.hip_center) * sc
            raw  = dp / dt
            self._spd_win.append(raw)
            fm.speed            = float(np.mean(self._spd_win))
            fm.body_center_disp = dp
            if len(self.pose_frames) >= 3:
                p2  = self.pose_frames[-3]
                dp2 = dist2d(prev.kp.hip_center, p2.kp.hip_center) * sc
                dt2 = prev.timestamp - p2.timestamp + 1e-9
                fm.acceleration = (raw - dp2 / dt2) / dt

        if len(self.pose_frames) >= 5:
            pos  = [p.kp.hip_center for p in list(self.pose_frames)[-5:]]
            vecs = [(pos[i+1][0]-pos[i][0], pos[i+1][1]-pos[i][1]) for i in range(4)]
            for i in range(len(vecs) - 1):
                v1, v2 = np.array(vecs[i]), np.array(vecs[i+1])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 2 and n2 > 2 and math.acos(
                        np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)) > math.radians(28):
                    fm.direction_change = True

        MASS_KG = getattr(self, 'player_mass_kg', 75.0)
        G = 9.81
        Cr = 4.0
        v = max(fm.speed, 0.1)
        a = fm.acceleration
        P_loco  = Cr * v * MASS_KG
        g_eq    = max(0., a) / G
        P_accel = Cr * g_eq * v * MASS_KG
        fm.energy_expenditure = P_loco + P_accel + 80.0

        ks = sum((155 - ang) / 155 for ang in
                 [fm.left_knee_angle, fm.right_knee_angle] if ang < 155)
        fm.joint_stress = min(1., ks / 2)
        ls  = clamp01(fm.trunk_lean / 25.) * (max(0, fm.speed - 1.0) / 5.0)
        asym = clamp01(abs(fm.left_knee_angle - fm.right_knee_angle) / 40.)
        fm.joint_stress = clamp01(fm.joint_stress * 0.5 + ls * 0.3 + asym * 0.2)

        if len(self._spd_win) >= 10:
            s = list(self._spd_win)
            fm.fatigue_index = max(0., min(1., (np.mean(s[:5]) - np.mean(s[-5:])) /
                                           (np.mean(s[:5]) + 1e-6)))

        valgus_deg  = (abs(fm.l_valgus_clinical) + abs(fm.r_valgus_clinical)) / 2.0
        p_valgus    = clamp01(valgus_deg / 15.0)
        p_knee_asym = clamp01(abs(fm.left_knee_angle - fm.right_knee_angle) / 30.)
        p_accel     = clamp01(abs(fm.acceleration) / 12.)
        fm.injury_risk = 0.45 * p_valgus + 0.30 * p_knee_asym + 0.25 * p_accel

        p_trunk    = clamp01(fm.trunk_lean / 30.)
        cumulative = 0.40 * fm.joint_stress + 0.35 * p_trunk + 0.25 * fm.fatigue_index

        raw_risk = (0.60 * fm.injury_risk + 0.40 * cumulative) * fm.perspective_confidence
        self._risk_win.append(raw_risk)
        fm.risk_score = float(np.mean(self._risk_win)) * 100.
        return fm

    # ── Post-processing ───────────────────────────────────────────────────────

    def _post_gait(self, fps: float):
        if len(self.pose_frames) < 15:
            return
        sc = self.PIX_TO_M or 0.002
        la = smooth_arr([p.kp.left_ankle[1]  for p in self.pose_frames])
        ra = smooth_arr([p.kp.right_ankle[1] for p in self.pose_frames])
        md = max(5, int(fps * .18))

        if HAS_SCIPY:
            lp, _ = find_peaks( la, distance=md, prominence=2)
            rp, _ = find_peaks( ra, distance=md, prominence=2)
        else:
            def pk(arr, d):
                pks = []
                for i in range(1, len(arr) - 1):
                    if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                        if not pks or i - pks[-1] >= d:
                            pks.append(i)
                return np.array(pks)
            lp = pk(la, md)
            rp = pk(ra, md)

        pos  = [p.kp.hip_center for p in self.pose_frames]
        strl, stt, flt = [], [], []
        for peaks in [lp, rp]:
            for i in range(1, len(peaks)):
                i0, i1 = peaks[i-1], peaks[i]
                if i1 >= len(pos):
                    continue
                sl = dist2d(pos[i0], pos[i1]) * sc
                if .15 < sl < 3.5:
                    strl.append(sl)
                st = (i1 - i0) / fps
                if .08 < st < 2.0:
                    stt.append(st)
                    flt.append(max(0., st * .35))

        n = min(len(lp), len(rp)) - 1
        if n > 0:
            li = [(lp[i+1] - lp[i]) / fps for i in range(n)]
            ri = [(rp[i+1] - rp[i]) / fps for i in range(n)]
            m  = min(len(li), len(ri))
            sym = float(np.mean(
                [1 - abs(l - r) / (l + r + 1e-9) for l, r in zip(li[:m], ri[:m])]
            )) * 100
        else:
            sym = 94.

        sv   = float(np.std(strl) / (np.mean(strl) + 1e-9) * 100) if len(strl) > 2 else 3.5
        asl  = float(np.mean(strl)) if strl else 1.35
        ast  = float(np.mean(stt))  if stt  else .38
        aft  = float(np.mean(flt))  if flt  else .13
        acad = 60. / ast if ast > 0 else 158.

        for fm in self.frame_metrics:
            fm.stride_length   = asl
            fm.step_time       = ast
            fm.flight_time     = aft
            fm.cadence         = acad
            fm.gait_symmetry   = sym
            fm.stride_variability = sv

        hip_x   = [p.kp.hip_center[0] for p in self.pose_frames]
        lat_bal = clamp01(
            np.std(hip_x) / (max(1, np.mean([p.bbox[2] for p in self.pose_frames])) * 0.1)
            if hip_x else 0.
        )
        for fm in self.frame_metrics:
            sr = max(0., (100 - fm.gait_symmetry) / 100)
            vr = min(1., fm.stride_variability / 25)
            lr = min(1., fm.trunk_lean / 40)
            fm.fall_risk    = clamp01(sr * .3 + vr * .2 + lr * .2 + lat_bal * .3)
            ar              = min(1., abs(fm.acceleration) / 12)
            fm.injury_risk  = fm.joint_stress * .5 + ar * .3 + fm.fatigue_index * .2

    def _build_summary(self):
        if not self.frame_metrics:
            return
        fms = self.frame_metrics
        s   = self.summary
        sc  = self.PIX_TO_M or 0.002

        s.total_frames    = len(fms)
        s.duration_seconds = fms[-1].timestamp + (1.0 / (self._fps_cache or 30.))

        spds = np.array([f.speed for f in fms])
        s.avg_speed = float(np.mean(spds))
        s.max_speed = float(np.max(spds))

        def anz(a):
            v = [getattr(f, a) for f in fms if getattr(f, a) > 0]
            return float(np.mean(v)) if v else 0.

        s.avg_stride_length        = anz("stride_length")
        s.avg_step_time            = anz("step_time")
        s.avg_cadence              = anz("cadence")  # strides per minute
        s.avg_flight_time          = anz("flight_time")
        s.estimated_energy_kcal_hr = float(np.mean([f.energy_expenditure for f in fms]))
        s.gait_symmetry_pct        = float(np.mean([f.gait_symmetry for f in fms]))
        s.stride_variability_pct   = float(np.mean([f.stride_variability for f in fms]))

        dc = sum(1 for f in fms if f.direction_change)
        s.direction_change_freq = dc / max(s.duration_seconds / 60, 1e-6)
        s.peak_risk_score = float(np.max([f.risk_score for f in fms]))

        if len(self.pose_frames) >= 2:
            s.total_distance_m = sum(
                dist2d(self.pose_frames[i].kp.hip_center, self.pose_frames[i-1].kp.hip_center) * sc
                for i in range(1, len(self.pose_frames))
            )

        def rl(a):
            return self._risk_label(float(np.mean([getattr(f, a) for f in fms])))

        s.fall_risk_label    = rl("fall_risk")
        s.injury_risk_label  = rl("injury_risk")
        s.body_stress_label  = rl("joint_stress")
        s.fatigue_label      = rl("fatigue_index")

        avg_valgus = float(np.mean(
            [abs(f.l_valgus_clinical) + abs(f.r_valgus_clinical) for f in fms]
        )) / 2.
        ai = float(np.mean([f.injury_risk for f in fms]))
        if avg_valgus > 10.:
            s.injury_risk_detail = "valgus collapse detected (>10°)"
        elif ai > .5:
            s.injury_risk_detail = "high knee load / acceleration stress"
        elif ai > .3:
            s.injury_risk_detail = "moderate joint stress"
        else:
            s.injury_risk_detail = "within normal range"

        if self.bio_engine:
            bd = self.bio_engine.summary_dict()
            s.double_support_pct = bd.get("double_support_pct", 0.0)
            avg_pr = bd.get("pelvis_rotation_mean", 0.0)
            # If explicit rotation is missing, estimate from trunk lean as fallback
            if avg_pr == 0:
                avg_pr = float(np.mean([f.trunk_lean for f in fms])) * 0.4
            s.avg_pelvic_rotation = avg_pr

    @staticmethod
    def _risk_label(v) -> str:
        return "Low" if v < .25 else "Moderate" if v < .55 else "High"

    # ── Sports2D native pipeline ──────────────────────────────────────────────

    def run_sports2d(self, result_dir: str,
                     mode: str = "balanced",
                     show_realtime: bool = False,
                     person_ordering: str = "greatest_displacement",
                     do_ik: bool = False,
                     use_augmentation: bool = False,
                     visible_side: str = "auto front",
                     participant_mass_kg: float = 75.0) -> dict:
        """
        Run Sports2D on the video.  This is always the first step — its
        on_click picker IS the player selection mechanism when --pick is used.
        After Sports2D finishes, we seed the custom tracker from the TRC data
        so both pipelines analyse the exact same player.
        """
        self.sports2d_runner = Sports2DRunner(
            video_path          = self.video_path,
            result_dir          = result_dir,
            player_height_m     = self.player_height_m,
            participant_mass_kg = participant_mass_kg,
            mode                = mode,
            show_realtime       = show_realtime,
            person_ordering     = person_ordering,
            do_ik               = do_ik,
            use_augmentation    = use_augmentation,
            visible_side        = visible_side,
        )
        outputs = self.sports2d_runner.run()

        # ── Seed our custom tracker from Sports2D's TRC output ───────────────
        # This guarantees both pipelines follow the same player.
        seed = self.sports2d_runner.get_seed_from_trc()
        if seed is not None:
            self.lock = TargetLock(
                seed["seed_bbox"], seed["hist"], seed["seed_frame"]
            )
            print("[S2D] Custom tracker seeded from Sports2D TRC data.")
        else:
            print("[S2D] Could not seed from TRC — custom tracker uses original pick.")

        return outputs

    # ── Unified export ────────────────────────────────────────────────────────

    def export_unified(self, json_path: str, csv_path: str,
                       trc_path: Optional[str] = None,
                       mot_path: Optional[str] = None):
        """
        Consolidate ALL data into two unified files:
          - data_output.json : hierarchical structured data
          - bio_metrics.csv  : flat time-series for analysis

        Optionally writes OpenSim-compatible .trc and .mot files.
        """
        # ── Build unified per-frame records ───────────────────────────────────
        bio_by_frame: dict = {}
        if self.bio_engine and self.bio_engine.frames:
            for bf in self.bio_engine.frames:
                bio_by_frame[bf.frame_idx] = asdict(bf)

        unified_frames = []
        for fm in self.frame_metrics:
            record = asdict(fm)
            bio = bio_by_frame.get(fm.frame_idx, {})
            # Merge bio fields — prefix with "bio_" to avoid name collision
            for k, v in bio.items():
                if k not in ("frame_idx", "timestamp"):
                    record[f"bio_{k}"] = v

            # Append Sports2D keypoints if available (from TRC data)
            unified_frames.append(record)

        # ── Sports2D angle summary ────────────────────────────────────────────
        s2d_angle_summary: dict = {}
        s2d_pose_summary:  dict = {}
        if self.sports2d_runner:
            mot_df = self.sports2d_runner.load_mot_angles()
            if mot_df is not None and not mot_df.empty:
                angle_cols = [c for c in mot_df.columns if c.lower() != "time"]
                for col in angle_cols:
                    try:
                        vals = pd.to_numeric(mot_df[col], errors="coerce").dropna()
                        if len(vals):
                            s2d_angle_summary[col] = {
                                "mean": float(vals.mean()),
                                "max":  float(vals.max()),
                                "min":  float(vals.min()),
                                "std":  float(vals.std()),
                            }
                    except Exception:
                        pass
            trc_df = self.sports2d_runner.load_trc_pose(metres=True)
            if trc_df is not None and not trc_df.empty:
                s2d_pose_summary["trc_shape"] = list(trc_df.shape)
                s2d_pose_summary["trc_columns"] = list(trc_df.columns)

        # ── JSON — hierarchical ───────────────────────────────────────────────
        payload = {
            "metadata": {
                "player_id":   self.player_id,
                "video_path":  self.video_path,
                "fps":         self._fps_cache,
                "pix_to_m":   self.PIX_TO_M,
                "total_frames": len(self.frame_metrics),
                "angle_backend": "sports2d" if HAS_SPORTS2D else "scipy" if HAS_SCIPY else "numpy",
            },
            "player_summary":   asdict(self.summary),
            "biomechanics_summary": self.bio_engine.summary_dict() if self.bio_engine else {},
            "sports2d_angle_summary": s2d_angle_summary,
            "sports2d_pose_summary":  s2d_pose_summary,
            "sports2d_output_files":  self.sports2d_runner.outputs if self.sports2d_runner else {},
            "frames": unified_frames,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[EXPORT] data_output.json → {json_path}  ({len(unified_frames)} frames)")

        # ── CSV — flat time-series ────────────────────────────────────────────
        df = pd.DataFrame(unified_frames)

        # Merge Sports2D MOT angles as additional columns if available
        if self.sports2d_runner:
            mot_df = self.sports2d_runner.load_mot_angles()
            if mot_df is not None and not mot_df.empty and "time" in mot_df.columns:
                mot_df = mot_df.rename(columns={"time": "timestamp"})
                mot_df.columns = ["s2d_" + c if c != "timestamp" else c for c in mot_df.columns]
                df = pd.merge(df, mot_df, on="timestamp", how="left")

        df.to_csv(csv_path, index=False)
        print(f"[EXPORT] bio_metrics.csv → {csv_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")

        # ── OpenSim TRC ───────────────────────────────────────────────────────
        if trc_path and self.pose_frames:
            writer = OpenSimFileWriter()
            writer.write_trc(
                pose_frames     = self.pose_frames,
                path            = trc_path,
                fps             = self._fps_cache,
                pix_to_m        = self.PIX_TO_M or 0.002,
                frame_height_px = self._frame_height_px or 720,
            )

        # ── OpenSim MOT ───────────────────────────────────────────────────────
        if mot_path and self.bio_engine and self.bio_engine.frames:
            writer = OpenSimFileWriter()
            writer.write_mot(
                bio_frames = self.bio_engine.frames,
                path       = mot_path,
                fps        = self._fps_cache,
            )

        return payload

    # ── Legacy export helpers (kept for backward compatibility) ───────────────

    def export_json(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "player_summary": asdict(self.summary),
                "frame_metrics":  [asdict(m) for m in self.frame_metrics],
            }, f, indent=2)
        print(f"[EXPORT] JSON → {path}")

    def export_csv(self, path: str):
        pd.DataFrame([asdict(m) for m in self.frame_metrics]).to_csv(path, index=False)
        print(f"[EXPORT] CSV  → {path}")

    def export_biomechanics_csv(self, path: str):
        if self.bio_engine and self.bio_engine.frames:
            self.bio_engine.get_dataframe().to_csv(path, index=False)
            print(f"[EXPORT] Bio CSV → {path}")

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(m) for m in self.frame_metrics])

    # ── Report ────────────────────────────────────────────────────────────────

    def get_report_string(self) -> str:
        s   = self.summary
        dm  = self._det_layer.mode.upper() if hasattr(self, '_det_layer') and self._det_layer else "BLOB"
        bio = "sports2d" if HAS_SPORTS2D else "scipy" if HAS_SCIPY else "numpy"
        W   = 70
        lines = ["=" * W,
                 f"SPORTS ANALYTICS v6 — Player #{s.player_id} [{dm}]".center(W),
                 "=" * W, "",
                 "SESSION OVERVIEW", "-" * W,
                 f"  Duration        : {s.duration_seconds:>6.1f} s",
                 f"  Total Frames    : {s.total_frames:>6}",
                 f"  Total Distance  : {s.total_distance_m:>6.1f} m",
                 f"  Angle Backend   : {bio}", "",
                 "PLAYER METRICS", "-" * W,
                 f"  Avg Speed       : {s.avg_speed:>6.2f} m/s",
                 f"  Max Speed       : {s.max_speed:>6.2f} m/s",
                 f"  Avg Stride      : {s.avg_stride_length:>6.2f} m",
                 f"  Avg Cadence     : {s.avg_cadence:>6.0f} strides/min",
                 f"  Avg Step Time   : {s.avg_step_time:>6.2f} s",
                 f"  Changes/Min     : {s.direction_change_freq:>6.1f}",
                 f"  Energy (avg)    : {s.estimated_energy_kcal_hr:>6.0f} W"]

        if self.bio_engine and self.bio_engine.frames:
            bd = self.bio_engine.summary_dict()
            lines += ["", "BIOMECHANICS  (Butterworth 6 Hz)", "-" * W,
                      f"  L Knee flexion  : {bd.get('left_knee_flexion_mean',0):>6.1f}° avg  {bd.get('left_knee_flexion_std',0):.1f}° sd",
                      f"  R Knee flexion  : {bd.get('right_knee_flexion_mean',0):>6.1f}° avg  {bd.get('right_knee_flexion_std',0):.1f}° sd",
                      f"  L Hip flexion   : {bd.get('left_hip_flexion_mean',0):>6.1f}°",
                      f"  R Hip flexion   : {bd.get('right_hip_flexion_mean',0):>6.1f}°",
                      f"  L Ankle dorsi   : {bd.get('left_ankle_dorsiflexion_mean',0):>6.1f}°",
                      f"  R Ankle dorsi   : {bd.get('right_ankle_dorsiflexion_mean',0):>6.1f}°",
                      f"  Trunk lat lean  : {bd.get('trunk_lateral_lean_mean',0):>6.1f}°",
                      f"  Pelvis obliquity: {bd.get('pelvis_obliquity_mean',0):>6.1f}°",
                      f"  Arm swing asym  : {bd.get('arm_swing_asymmetry_mean',0):>6.1f}°",
                      f"  Double support  : {bd.get('double_support_pct',0):>6.1f}%",
                      f"  Heel strikes L/R: {bd.get('lhs_count',0)} / {bd.get('rhs_count',0)}"]
            lvc = bd.get('left_valgus_clinical_mean',  0)
            rvc = bd.get('right_valgus_clinical_mean', 0)
            lines.append(f"  L Valgus (clin) : {lvc:>+6.1f}°{'  ⚠ VALGUS' if abs(lvc)>10 else ''}")
            lines.append(f"  R Valgus (clin) : {rvc:>+6.1f}°{'  ⚠ VALGUS' if abs(rvc)>10 else ''}")

        lines += ["", "RISK INDICATORS", "-" * W,
                  f"  Peak Risk Score : {s.peak_risk_score:>6.0f} / 100",
                  f"  Gait Symmetry   : {s.gait_symmetry_pct:>6.1f} %",
                  f"  Acute Inj. Risk : {s.injury_risk_label}",
                  f"  Body Stress     : {s.body_stress_label}",
                  f"  Fatigue Level   : {s.fatigue_label}",
                  f"  Risk Detail     : {s.injury_risk_detail}",
                  "", "=" * W]
        return "\n".join(lines)
