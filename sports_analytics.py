"""
Juventus Sports Analytics System  v3
======================================
Full pipeline:
  1. Pre-scan  — find the most persistent player across video
  2. Tracking  — MIL tracker + MOG2 blob verification
  3. Pose      — contour-fitted anthropometric model with per-joint
                 Kalman smoothing (fluid, jitter-free motion)
  4. Biomechanics — speed, stride, cadence, angles, energy, valgus
  5. Risk      — rolling-window injury / fall / fatigue scoring
  6. Rendering — gradient bones, glow joints, animated risk gauge,
                 full-frame HUD dashboard
  7. Export    — annotated MP4, JSON, CSV
"""

import cv2
import numpy as np
import pandas as pd
import json
import math
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

JOINT_NAMES = [
    "head","neck",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
    "left_foot","right_foot",
    "hip_center","shoulder_center",
]

@dataclass
class PoseKeypoints:
    head:            Tuple[float,float] = (0.,0.)
    neck:            Tuple[float,float] = (0.,0.)
    left_shoulder:   Tuple[float,float] = (0.,0.)
    right_shoulder:  Tuple[float,float] = (0.,0.)
    left_elbow:      Tuple[float,float] = (0.,0.)
    right_elbow:     Tuple[float,float] = (0.,0.)
    left_wrist:      Tuple[float,float] = (0.,0.)
    right_wrist:     Tuple[float,float] = (0.,0.)
    left_hip:        Tuple[float,float] = (0.,0.)
    right_hip:       Tuple[float,float] = (0.,0.)
    left_knee:       Tuple[float,float] = (0.,0.)
    right_knee:      Tuple[float,float] = (0.,0.)
    left_ankle:      Tuple[float,float] = (0.,0.)
    right_ankle:     Tuple[float,float] = (0.,0.)
    left_foot:       Tuple[float,float] = (0.,0.)
    right_foot:      Tuple[float,float] = (0.,0.)
    hip_center:      Tuple[float,float] = (0.,0.)
    shoulder_center: Tuple[float,float] = (0.,0.)


@dataclass
class PoseFrame:
    frame_idx: int
    timestamp: float
    bbox:      Tuple[int,int,int,int]
    kp:        PoseKeypoints


@dataclass
class FrameMetrics:
    frame_idx:         int   = 0
    timestamp:         float = 0.
    speed:             float = 0.
    acceleration:      float = 0.
    stride_length:     float = 0.
    step_time:         float = 0.
    cadence:           float = 0.
    flight_time:       float = 0.
    left_knee_angle:   float = 0.
    right_knee_angle:  float = 0.
    left_hip_angle:    float = 0.
    right_hip_angle:   float = 0.
    trunk_lean:        float = 0.
    direction_change:  bool  = False
    energy_expenditure:float = 0.
    gait_symmetry:     float = 100.
    stride_variability:float = 0.
    fall_risk:         float = 0.
    injury_risk:       float = 0.
    joint_stress:      float = 0.
    fatigue_index:     float = 0.
    body_center_disp:  float = 0.
    l_valgus:          float = 0.
    r_valgus:          float = 0.
    risk_score:        float = 0.   # 0-100 composite


@dataclass
class PlayerSummary:
    player_id:                int   = 1
    total_frames:             int   = 0
    duration_seconds:         float = 0.
    avg_speed:                float = 0.
    max_speed:                float = 0.
    avg_stride_length:        float = 0.
    avg_step_time:            float = 0.
    avg_cadence:              float = 0.
    avg_flight_time:          float = 0.
    direction_change_freq:    float = 0.
    estimated_energy_kcal_hr: float = 0.
    gait_symmetry_pct:        float = 0.
    stride_variability_pct:   float = 0.
    total_distance_m:         float = 0.
    peak_risk_score:          float = 0.
    fall_risk_label:          str   = "Low"
    injury_risk_label:        str   = "Low"
    injury_risk_detail:       str   = ""
    body_stress_label:        str   = "Low"
    fatigue_label:            str   = "Low"


# ══════════════════════════════════════════════════════════════════════════════
#  MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def angle_3pts(a, b, c) -> float:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    n  = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc)/n, -1, 1))))

def dist2d(p1, p2) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def smooth_arr(arr, w=5):
    return uniform_filter1d(np.array(arr, dtype=float), size=w)

def clamp01(x): return float(np.clip(x, 0., 1.))

def lerp_color(c1, c2, t):
    """Linearly interpolate two BGR colours."""
    t = clamp01(t)
    return tuple(int(c1[i]*(1-t) + c2[i]*t) for i in range(3))

def risk_color(score_0_100):
    """Green → Yellow → Red."""
    t = clamp01(score_0_100 / 100.)
    if t < 0.5:
        return lerp_color((0,200,0), (0,200,255), t*2)
    else:
        return lerp_color((0,200,255), (0,0,230), (t-0.5)*2)


# ══════════════════════════════════════════════════════════════════════════════
#  PER-JOINT KALMAN SMOOTHER
#  Keeps a 1-D position+velocity Kalman filter for each joint coordinate.
#  This gives smooth, physically plausible motion even if the bbox jitters.
# ══════════════════════════════════════════════════════════════════════════════

class JointKalman:
    """Scalar Kalman filter for one joint coordinate (x or y)."""
    def __init__(self, process_noise=1.5, obs_noise=8.0):
        self.x  = None   # state: position
        self.v  = 0.     # state: velocity
        self.P  = np.array([[100.,0.],[0.,100.]])
        self.Q  = np.diag([process_noise, process_noise*2])
        self.R  = obs_noise
        self.F  = np.array([[1.,1.],[0.,1.]])
        self.H  = np.array([[1.,0.]])

    def update(self, z: float) -> float:
        if self.x is None:
            self.x = z; return z
        # Predict
        state  = np.array([self.x, self.v])
        state  = self.F @ state
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # Update
        y  = z - (self.H @ state)[0]
        S  = (self.H @ P_pred @ self.H.T)[0,0] + self.R
        K  = (P_pred @ self.H.T) / S
        state = state + (K * y).flatten()
        self.P = (np.eye(2) - np.outer(K.flatten(), self.H)) @ P_pred
        self.x, self.v = float(state[0]), float(state[1])
        return self.x


class PoseKalmanSmoother:
    """One JointKalman per joint, per axis."""
    def __init__(self):
        self._kx: dict[str, JointKalman] = {}
        self._ky: dict[str, JointKalman] = {}

    def smooth(self, kp: PoseKeypoints) -> PoseKeypoints:
        out = PoseKeypoints()
        for name in JOINT_NAMES:
            raw = getattr(kp, name)
            if name not in self._kx:
                self._kx[name] = JointKalman()
                self._ky[name] = JointKalman()
            sx = self._kx[name].update(raw[0])
            sy = self._ky[name].update(raw[1])
            object.__setattr__(out, name, (sx, sy))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  POSE ESTIMATOR
#  Uses the player contour mask (not just the bbox rectangle) to fit joint
#  positions anatomically — shoulders snap to the widest upper-body point,
#  waist to the narrowest, hips to the widest lower-body point, etc.
#  Limb swing is driven by the actual horizontal displacement of the bbox
#  centre, giving motion-reactive gait rather than a fixed-period oscillator.
# ══════════════════════════════════════════════════════════════════════════════

class ContourPoseEstimator:
    """
    Fit 18 body keypoints to the player silhouette.
    Falls back to anthropometric proportions when the contour is unavailable.
    """

    # Anthropometric vertical proportions (fraction of bbox height, top=0)
    _VP = dict(head=0.04, neck=0.11, shoulder=0.20, elbow=0.34,
               wrist=0.46, hip=0.54, knee=0.73, ankle=0.91, foot=0.99)

    def __init__(self):
        self._prev_cx     = None   # previous bbox centre-x for swing detection
        self._displacement_history = deque(maxlen=8)

    def estimate(self, frame, bbox, timestamp: float,
                 running_speed: float = 0.) -> PoseKeypoints:
        x, y, w, h = bbox
        cx = x + w / 2.

        # ── Measure actual lateral displacement this frame ─────────────────
        disp = 0.
        if self._prev_cx is not None:
            disp = cx - self._prev_cx
        self._prev_cx = cx
        self._displacement_history.append(abs(disp))
        motion_mag = float(np.mean(self._displacement_history)) if self._displacement_history else 0.

        # ── Try to fit shoulders / waist / hips from the contour ──────────
        col_widths = self._column_widths(frame, bbox)
        shoulder_x_half, hip_x_half = self._fit_body_widths(col_widths, w, h)

        # ── Gait phase from accumulated displacement ───────────────────────
        # Use distance travelled (pixels) to drive a continuous phase,
        # so the skeleton actually syncs with the person's gait.
        dist_sum = sum(self._displacement_history)
        phase = (dist_sum / max(w * 0.18, 4.)) * math.pi  # ~half-step per stride_width/18

        # Swing amplitude: scales with speed
        swing = clamp01(running_speed / 9.)
        arm_sw = swing * 0.10 * w
        leg_sw = swing * 0.08 * w
        k_lift = swing * 0.08 * h

        def vy(f): return y + f * h

        kp = PoseKeypoints()

        # Head & neck
        kp.head  = (cx, vy(self._VP["head"]))
        kp.neck  = (cx, vy(self._VP["neck"]))

        # Shoulders
        ls = (cx - shoulder_x_half, vy(self._VP["shoulder"]))
        rs = (cx + shoulder_x_half, vy(self._VP["shoulder"]))
        kp.left_shoulder  = ls
        kp.right_shoulder = rs
        kp.shoulder_center = ((ls[0]+rs[0])/2., (ls[1]+rs[1])/2.)

        # Arms — opposite phase to legs
        aoff = arm_sw * math.sin(phase)
        le   = (ls[0] - aoff, vy(self._VP["elbow"]))
        re   = (rs[0] + aoff, vy(self._VP["elbow"]))
        kp.left_elbow  = le
        kp.right_elbow = re
        kp.left_wrist  = (le[0] - aoff * 0.55, vy(self._VP["wrist"]))
        kp.right_wrist = (re[0] + aoff * 0.55, vy(self._VP["wrist"]))

        # Hips
        lh = (cx - hip_x_half, vy(self._VP["hip"]))
        rh = (cx + hip_x_half, vy(self._VP["hip"]))
        kp.left_hip   = lh
        kp.right_hip  = rh
        kp.hip_center = ((lh[0]+rh[0])/2., (lh[1]+rh[1])/2.)

        # Legs
        loff = leg_sw * math.sin(phase)
        roff = -loff
        ll   = k_lift * max(0., math.sin(phase))
        rl   = k_lift * max(0., -math.sin(phase))

        kp.left_knee  = (lh[0] + loff, vy(self._VP["knee"])  - ll)
        kp.right_knee = (rh[0] + roff, vy(self._VP["knee"])  - rl)
        kp.left_ankle = (lh[0] + loff * 0.45, vy(self._VP["ankle"]) - ll * 0.5)
        kp.right_ankle= (rh[0] + roff * 0.45, vy(self._VP["ankle"]) - rl * 0.5)
        kp.left_foot  = (kp.left_ankle[0]  + w * 0.07, vy(self._VP["foot"]))
        kp.right_foot = (kp.right_ankle[0] + w * 0.07, vy(self._VP["foot"]))

        return kp

    # ── Contour fitting helpers ────────────────────────────────────────────

    def _column_widths(self, frame, bbox) -> Optional[np.ndarray]:
        """Return horizontal width of the player mask at each scanline."""
        bx, by, bw, bh = bbox
        H, W = frame.shape[:2]
        bx2 = min(bx+bw, W); by2 = min(by+bh, H)
        bx  = max(0, bx);    by  = max(0, by)
        if bx2-bx < 5 or by2-by < 5:
            return None
        crop = frame[by:by2, bx:bx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        widths = np.array([np.sum(mask[r] > 0) for r in range(mask.shape[0])],
                          dtype=float)
        # smooth to remove noise
        return smooth_arr(widths, w=max(3, bh//20)) if len(widths) > 5 else None

    def _fit_body_widths(self, col_widths, bw, bh):
        """Infer shoulder_half and hip_half widths from the silhouette profile."""
        default_sh  = bw * 0.29
        default_hip = bw * 0.17
        if col_widths is None or len(col_widths) < 10:
            return default_sh, default_hip

        n = len(col_widths)
        # Upper body: rows 15%–40% of height
        upper = col_widths[int(n*0.15):int(n*0.40)]
        # Lower body: rows 48%–68% of height
        lower = col_widths[int(n*0.48):int(n*0.68)]

        sh  = float(np.max(upper)) / 2. if len(upper) else default_sh
        hip = float(np.max(lower)) / 2. if len(lower) else default_hip

        # Clamp to sane range
        sh  = float(np.clip(sh,  bw*0.18, bw*0.42))
        hip = float(np.clip(hip, bw*0.10, bw*0.32))
        return sh, hip


# ══════════════════════════════════════════════════════════════════════════════
#  BLOB DETECTOR  (shared utility)
# ══════════════════════════════════════════════════════════════════════════════

def get_human_blobs(frame, bg_sub, min_area=2500, max_area=80000):
    """
    Detect human-shaped motion blobs with strict quality filters:
      - Tighter aspect ratio (1.4-4.5): real standing humans only
      - Minimum fill ratio > 0.28: rejects shadows, fence fragments, L-shapes
      - Non-maximum suppression (IoU > 0.35): merges split detections of the
        same person so one player never appears as two separate blobs
    """
    mask = bg_sub.apply(frame)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = bh / (bw + 1e-6)
        if not (1.4 <= aspect <= 4.5):
            continue
        fill = area / (bw * bh + 1e-6)
        if fill < 0.28:
            continue
        candidates.append((bx, by, bw, bh, area))
    if not candidates:
        return []
    # NMS: sort largest first, suppress boxes that overlap heavily
    candidates.sort(key=lambda c: c[4], reverse=True)
    kept, suppressed = [], set()
    for i, ci in enumerate(candidates):
        if i in suppressed:
            continue
        kept.append(ci[:4])
        for j, cj in enumerate(candidates):
            if j <= i or j in suppressed:
                continue
            if blob_iou(ci[:4], cj[:4]) > 0.35:
                suppressed.add(j)
    return kept

def blob_iou(a, b) -> float:
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    ix = max(0, min(ax+aw,bx+bw)-max(ax,bx))
    iy = max(0, min(ay+ah,by+bh)-max(ay,by))
    inter = ix*iy
    return inter/(aw*ah+bw*bh-inter+1e-6)

def crop_hist(frame, bbox):
    bx,by,bw,bh = bbox
    H,W = frame.shape[:2]
    bx,by=max(0,bx),max(0,by); bw=min(bw,W-bx); bh=min(bh,H-by)
    if bw<5 or bh<5: return None
    hsv  = cv2.cvtColor(frame[by:by+bh, bx:bx+bw], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[18,16],[0,180,0,256])
    cv2.normalize(hist, hist)
    return hist


# ══════════════════════════════════════════════════════════════════════════════
#  PRE-SCAN: find the most persistent player
# ══════════════════════════════════════════════════════════════════════════════

def select_primary_player(video_path: str, sample_step: int = 6,
                           min_area: int = 1200, max_area: int = 70000) -> Optional[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg    = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=36,
                                                detectShadows=False)
    tracks: List[dict] = []
    MAX_GAP = max(sample_step*5, 30)
    frame_idx = 0
    print(f"[PRE-SCAN] Scanning {total} frames (step={sample_step})…")

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % sample_step == 0:
            blobs = get_human_blobs(frame, bg, min_area, max_area)
            matched = set()
            for blob in blobs:
                bx,by,bw,bh = blob
                best_t, best_s = None, 0.
                for ti, tr in enumerate(tracks):
                    if ti in matched: continue
                    if frame_idx - tr["last_frame"] > MAX_GAP: continue
                    iou = blob_iou(blob, tr["last_bbox"])
                    rw,rh = tr["mean_size"]
                    size_sim = min(bw*bh,rw*rh)/(max(bw*bh,rw*rh)+1e-6)
                    score = iou*0.7+size_sim*0.3
                    if score>best_s and (iou>0.10 or size_sim>0.55):
                        best_s,best_t = score,ti
                h = crop_hist(frame, blob)
                if best_t is not None:
                    tr = tracks[best_t]; tr["frames"]+=1
                    if h is not None: tr["hists"].append(h)
                    n=tr["frames"]; pw,ph=tr["mean_size"]
                    tr["mean_size"]=((pw*(n-1)+bw)/n,(ph*(n-1)+bh)/n)
                    tr["last_bbox"]=blob; tr["last_frame"]=frame_idx
                    matched.add(best_t)
                else:
                    tracks.append({"frames":1,"hists":[h] if h is not None else [],
                                   "mean_size":(float(bw),float(bh)),
                                   "last_bbox":blob,"last_frame":frame_idx,
                                   "seed_bbox":blob,"seed_frame":frame_idx})
        frame_idx += 1
    cap.release()

    if not tracks: return None
    best = max(tracks, key=lambda t: t["frames"])
    print(f"[PRE-SCAN] Best track: {best['frames']} hits, seed={best['seed_bbox']}")
    mean_hist = None
    if best["hists"]:
        stacked = np.mean(best["hists"], axis=0).astype(np.float32)
        cv2.normalize(stacked, stacked); mean_hist = stacked
    return {"hist":mean_hist,"size":best["mean_size"],
            "seed_bbox":best["seed_bbox"],"seed_frame":best["seed_frame"]}


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PLAYER PICKER
#  Opens the first clear frame, shows all detected blobs numbered, and waits
#  for the user to click on the player they want to track.  Returns the same
#  primary_info dict that select_primary_player() returns.
# ══════════════════════════════════════════════════════════════════════════════

def pick_player_interactive(video_path: str,
                             min_area: int = 2500,
                             max_area: int = 80000) -> Optional[dict]:
    """
    Interactive player selector.
    Warms up MOG2, finds the frame with the most clean blobs, shows it
    in a window and waits for the user to click a player.
    Returns a primary_info dict for PlayerTracker.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bg    = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=36,
                                               detectShadows=False)

    # ── Warm-up: read frames and keep (frame, blobs, index) per sampled frame
    WARMUP = min(120, total // 3)
    candidates = []   # list of (frame_ndarray, blobs, frame_index)

    for fi in range(WARMUP):
        ret, frame = cap.read()
        if not ret:
            break
        blobs = get_human_blobs(frame, bg, min_area, max_area)
        if blobs:
            candidates.append((frame.copy(), blobs, fi))

    cap.release()

    if not candidates:
        print("[PICKER] No blobs detected — falling back to auto-select.")
        return select_primary_player(video_path)

    # Pick the entry with the most blobs (most players visible at once)
    best_frame, best_blobs, best_fi = max(candidates, key=lambda c: len(c[1]))

    # ── Build display ─────────────────────────────────────────────────────────
    display = best_frame.copy()
    # Slight darkening so highlights pop
    display = cv2.addWeighted(display, 0.65, np.zeros_like(display), 0.35, 0)

    COLORS = [
        (0,255,180),(0,140,255),(255,215,0),(0,200,255),
        (180,0,255),(0,255,80),(255,80,80),(80,255,255),
    ]
    for i, (bx, by, bw, bh) in enumerate(best_blobs):
        col = COLORS[i % len(COLORS)]
        # Thicker highlight box
        cv2.rectangle(display, (bx, by), (bx+bw, by+bh), col, 3, cv2.LINE_AA)
        # Number badge centred above box
        label = str(i + 1)
        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        bx_badge = bx + bw//2 - lw//2 - 6
        by_badge = max(0, by - 34)
        cv2.rectangle(display, (bx_badge, by_badge),
                      (bx_badge + lw + 12, by_badge + 28), col, -1)
        cv2.putText(display, label, (bx_badge + 6, by_badge + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Instruction banner
    BANNER_H = 52
    banner = np.full((BANNER_H, W, 3), 15, np.uint8)
    cv2.putText(banner,
                "CLICK the player to track   |   ESC = auto-select",
                (W//2 - 255, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 215, 0), 1, cv2.LINE_AA)
    display = np.vstack([banner, display])

    # ── Mouse callback ────────────────────────────────────────────────────────
    chosen = [None]   # will hold the selected (bx,by,bw,bh) tuple

    def on_click(event, cx, cy, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        adj_y = cy - BANNER_H
        if adj_y < 0:
            return
        # Check each blob
        for blob in best_blobs:
            bx, by, bw, bh = blob
            if bx <= cx <= bx + bw and by <= adj_y <= by + bh:
                chosen[0] = blob
                return
        # Outside all blobs — pick the nearest centre
        chosen[0] = min(best_blobs,
                        key=lambda b: math.hypot(cx - (b[0]+b[2]/2),
                                                 adj_y - (b[1]+b[3]/2)))

    cv2.namedWindow("Select Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Player", min(W, 1280), min(H + BANNER_H, 800))
    cv2.setMouseCallback("Select Player", on_click)

    print(f"\n[PICKER] {len(best_blobs)} player(s) detected.")
    print("[PICKER] Click on the player to track.  ESC = auto-select.\n")

    while True:
        cv2.imshow("Select Player", display)
        key = cv2.waitKey(20) & 0xFF
        if chosen[0] is not None or key == 27:
            break

    cv2.destroyAllWindows()

    if chosen[0] is None:
        print("[PICKER] ESC pressed — falling back to auto-select.")
        return select_primary_player(video_path)

    blob = chosen[0]
    bx, by, bw, bh = blob
    num  = best_blobs.index(blob) + 1
    hist = crop_hist(best_frame, blob)
    print(f"[PICKER] Player {num} selected — bbox={blob}")

    return {
        "hist":       hist,
        "size":       (float(bw), float(bh)),
        "seed_bbox":  blob,
        "seed_frame": best_fi,    # integer index, no numpy comparison needed
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-TARGET TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class PlayerTracker:
    _PENDING  = "pending"
    _TRACKING = "tracking"
    _LOST     = "lost"

    def __init__(self, primary_info: dict,
                 min_area=1200, max_area=70000,
                 iou_thr=0.12, hist_thr=0.38, patience=120):
        self.min_a=min_area; self.max_a=max_area
        self.iou_thr=iou_thr; self.hist_thr=hist_thr; self.patience=patience
        self._ref_hist  = primary_info["hist"]
        self._ref_size  = primary_info["size"]
        self._seed_bbox = primary_info["seed_bbox"]
        self._seed_frame= primary_info["seed_frame"]
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=300,varThreshold=36,detectShadows=False)
        self._state=self._PENDING; self._mil=None
        self._smooth_box=None; self._alpha=0.30
        self._lost_count=0; self._frame_idx=0

    def update(self, frame):
        blobs = get_human_blobs(frame, self.bg, self.min_a, self.max_a)
        if self._state==self._PENDING:   r=self._pending(frame,blobs)
        elif self._state==self._TRACKING:r=self._track(frame,blobs)
        else:                            r=self._reacquire(frame,blobs)
        self._frame_idx+=1; return r

    def _pending(self, frame, blobs):
        if self._frame_idx < self._seed_frame: return None
        best,bs = None,-1.
        for b in blobs:
            iou=blob_iou(b,self._seed_bbox); hs=self._hsim(frame,b)
            s=iou*0.5+hs*0.5
            if s>bs: bs,best=s,b
        if best is None and blobs:
            best=max(blobs,key=lambda b:self._hsim(frame,b))
        if best:
            self._init_mil(frame,best); self._upd_ref(frame,best)
            self._smooth_box=np.array(best,float)
            self._state=self._TRACKING
            print(f"[TRACKER] Locked at frame {self._frame_idx}, bbox={best}")
            return self._emit(best)
        return None

    def _track(self, frame, blobs):
        ok,mbox=self._mil.update(frame)
        mbox=tuple(int(v) for v in mbox) if ok else None
        best,bs=self._best_blob(mbox,blobs,frame)
        if best:
            self._init_mil(frame,best); self._upd_ref(frame,best)
            self._lost_count=0; return self._emit(best)
        if mbox and ok:
            self._lost_count=0; return self._emit(mbox)
        self._state=self._LOST; self._lost_count=0
        print(f"[TRACKER] Lost at frame {self._frame_idx}")
        return None

    def _reacquire(self, frame, blobs):
        self._lost_count+=1
        ht=max(0.22, self.hist_thr-self._lost_count*0.0008)
        best,bs=None,-1.
        for b in blobs:
            bx,by,bw,bh=b; rw,rh=self._ref_size
            ss=min(bw*bh,rw*rh)/(max(bw*bh,rw*rh)+1e-6)
            if ss<0.25: continue
            hs=self._hsim(frame,b); sc=hs*0.70+ss*0.30
            if hs>=ht and sc>bs: bs,best=sc,b
        if best:
            self._init_mil(frame,best); self._upd_ref(frame,best)
            self._smooth_box=np.array(best,float)
            self._state=self._TRACKING; self._lost_count=0
            print(f"[TRACKER] Re-acquired at frame {self._frame_idx}")
            return self._emit(best)
        if self._lost_count>=self.patience and blobs:
            fb=max(blobs,key=lambda b:self._hsim(frame,b))
            if self._hsim(frame,fb)>0.20:
                self._init_mil(frame,fb); self._smooth_box=np.array(fb,float)
                self._state=self._TRACKING; self._lost_count=0
                return self._emit(fb)
        return None

    def _best_blob(self, mbox, blobs, frame):
        if not blobs: return None,0.
        best,bs=None,0.
        for b in blobs:
            iou=blob_iou(mbox,b) if mbox else 0.
            hs=self._hsim(frame,b); sc=iou*0.60+hs*0.40
            if (iou>=self.iou_thr or hs>=self.hist_thr) and sc>bs:
                bs,best=sc,b
        return best,bs

    def _hsim(self, frame, bbox) -> float:
        if self._ref_hist is None: return 0.
        h=crop_hist(frame,bbox)
        if h is None: return 0.
        return float(cv2.compareHist(self._ref_hist,h,cv2.HISTCMP_CORREL))

    def _upd_ref(self, frame, bbox):
        h=crop_hist(frame,bbox)
        if h is not None:
            self._ref_hist=(0.85*(self._ref_hist if self._ref_hist is not None else h)
                            +0.15*h).astype(np.float32)
            cv2.normalize(self._ref_hist,self._ref_hist)
        bx,by,bw,bh=bbox; rw,rh=self._ref_size
        self._ref_size=(rw*0.9+bw*0.1, rh*0.9+bh*0.1)

    def _init_mil(self, frame, bbox):
        self._mil=cv2.TrackerMIL_create(); self._mil.init(frame,bbox)

    def _emit(self, bbox):
        arr=np.array(bbox,float)
        if self._smooth_box is None: self._smooth_box=arr
        else: self._smooth_box=self._alpha*arr+(1-self._alpha)*self._smooth_box
        return tuple(int(v) for v in self._smooth_box)

    @property
    def state(self): return self._state


# ══════════════════════════════════════════════════════════════════════════════
#  SKELETON RENDERER
#  — Gradient bone segments (colour transitions along each limb)
#  — Glow effect on joints (soft outer circle before hard inner fill)
#  — Per-limb colouring: left=cyan, right=orange, spine=white, at-risk=red
#  — Joints scale with confidence (bigger = more certain)
# ══════════════════════════════════════════════════════════════════════════════

# Connections: (joint_a, joint_b, color_a_BGR, color_b_BGR, thickness)
_C = (255,220,0)   # Juventus gold
_W = (240,240,240) # white
_L = (255,200,0)   # left  cyan-ish
_R = (0,140,255)   # right orange
_S = (180,240,180) # spine light green

BONE_DEFS = [
    # Spine
    ("head",          "neck",           _W,  _W,  4),
    ("neck",          "shoulder_center",_W,  _S,  4),
    ("shoulder_center","hip_center",    _S,  _S,  5),
    # Left arm
    ("left_shoulder", "left_elbow",     _L,  _L,  5),
    ("left_elbow",    "left_wrist",     _L,  _L,  4),
    # Right arm
    ("right_shoulder","right_elbow",    _R,  _R,  5),
    ("right_elbow",   "right_wrist",    _R,  _R,  4),
    # Shoulder girdle
    ("left_shoulder", "right_shoulder", _L,  _R,  4),
    # Hip girdle
    ("left_hip",      "right_hip",      _L,  _R,  5),
    # Left leg
    ("left_hip",      "left_knee",      _L,  _L,  7),
    ("left_knee",     "left_ankle",     _L,  _L,  6),
    ("left_ankle",    "left_foot",      _L,  _L,  4),
    # Right leg
    ("right_hip",     "right_knee",     _R,  _R,  7),
    ("right_knee",    "right_ankle",    _R,  _R,  6),
    ("right_ankle",   "right_foot",     _R,  _R,  4),
]


def draw_gradient_bone(img, p1, p2, c1, c2, thickness, risk_tint=0.):
    """Draw a bone segment with colour gradient and optional red risk tint."""
    steps = max(8, int(dist2d(p1,p2)/4))
    for i in range(steps):
        t  = i / max(steps-1, 1)
        t2 = (i+1) / max(steps-1, 1)
        # Interpolate colour along the bone
        col = lerp_color(c1, c2, t)
        # Blend toward red based on risk
        col = lerp_color(col, (0,0,220), risk_tint*0.6)
        px1 = (int(p1[0]+t*(p2[0]-p1[0])),   int(p1[1]+t*(p2[1]-p1[1])))
        px2 = (int(p1[0]+t2*(p2[0]-p1[0])),  int(p1[1]+t2*(p2[1]-p1[1])))
        cv2.line(img, px1, px2, col, thickness, cv2.LINE_AA)


def draw_glow_joint(img, pt, radius, color, glow_alpha=0.45):
    """Draw a joint with a soft glow halo then a filled centre."""
    px, py = int(pt[0]), int(pt[1])
    # Glow — draw several concentric translucent circles
    for r in range(radius+6, radius, -2):
        ov = img.copy()
        cv2.circle(ov, (px,py), r, color, -1, cv2.LINE_AA)
        alpha = glow_alpha * (1 - (r-radius)/6.)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    # Hard fill
    cv2.circle(img, (px,py), radius, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(img, (px,py), max(1,radius-2), color, -1, cv2.LINE_AA)


def render_skeleton(frame, kp: PoseKeypoints, risk_tint: float = 0.):
    """Full skeleton render with gradient bones + glowing joints."""
    kpd = {n: getattr(kp,n) for n in JOINT_NAMES}

    # 1. Draw bones
    for a, b, c1, c2, th in BONE_DEFS:
        if a in kpd and b in kpd:
            draw_gradient_bone(frame, kpd[a], kpd[b], c1, c2, th, risk_tint)

    # 2. Draw joints — larger for load-bearing ones
    joint_sizes = {
        "head":4, "neck":3,
        "left_shoulder":4,"right_shoulder":4,
        "left_elbow":3,"right_elbow":3,
        "left_wrist":3,"right_wrist":3,
        "left_hip":5,"right_hip":5,
        "left_knee":6,"right_knee":6,
        "left_ankle":5,"right_ankle":5,
        "left_foot":3,"right_foot":3,
    }
    for name, r in joint_sizes.items():
        if name in kpd:
            col = lerp_color(_L if "left" in name else _R if "right" in name else _W,
                             (0,0,220), risk_tint*0.5)
            draw_glow_joint(frame, kpd[name], r, col)


# ══════════════════════════════════════════════════════════════════════════════
#  RISK GAUGE  (animated arc drawn directly onto the frame)
# ══════════════════════════════════════════════════════════════════════════════

def draw_risk_gauge(frame, cx, cy, radius, score_0_100, label="RISK"):
    """Animated arc gauge showing risk 0–100."""
    bg_col  = (30,30,30)
    # Background arc
    cv2.ellipse(frame,(cx,cy),(radius,radius),225,0,270,bg_col,6,cv2.LINE_AA)
    # Filled arc
    sweep = int(270 * clamp01(score_0_100/100.))
    if sweep > 0:
        col = risk_color(score_0_100)
        cv2.ellipse(frame,(cx,cy),(radius,radius),225,0,sweep,col,6,cv2.LINE_AA)
    # Score text
    col = risk_color(score_0_100)
    cv2.putText(frame, f"{score_0_100:.0f}", (cx-18,cy+7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (cx-20,cy+radius-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class SportsAnalyzer:

    PIX_TO_M = None

    SKELETON_CONNECTIONS = [  # kept for legacy reference; rendering uses BONE_DEFS
        ("head","neck"),("neck","left_shoulder"),("neck","right_shoulder"),
        ("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
        ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
        ("left_shoulder","left_hip"),("right_shoulder","right_hip"),
        ("left_hip","right_hip"),
        ("left_hip","left_knee"),("left_knee","left_ankle"),("left_ankle","left_foot"),
        ("right_hip","right_knee"),("right_knee","right_ankle"),("right_ankle","right_foot"),
    ]

    def __init__(self, video_path, output_video_path="output_annotated.mp4",
                 player_id=1, fps_override=None, pick=False):
        """
        pick=False  →  auto-select the most persistent player (default)
        pick=True   →  open an interactive window so you click the player
        """
        self.video_path        = video_path
        self.output_video_path = output_video_path
        self.player_id         = player_id
        self.fps_override      = fps_override
        self.pose_est          = ContourPoseEstimator()
        self.smoother          = PoseKalmanSmoother()
        self.pose_frames:   List[PoseFrame]    = []
        self.frame_metrics: List[FrameMetrics] = []
        self.summary = PlayerSummary(player_id=player_id)
        self._spd_win   = deque(maxlen=30)
        self._risk_win  = deque(maxlen=15)

        if pick:
            print("[INFO] Interactive player selection…")
            primary = pick_player_interactive(video_path)
        else:
            print("[INFO] Running pre-scan to identify primary player…")
            primary = select_primary_player(video_path)

        if primary is None:
            raise RuntimeError("Could not identify any player candidates.")
        self.tracker = PlayerTracker(primary)

        # ── Wow-factor state ──────────────────────────────────────────────────
        self._trail: deque = deque(maxlen=60)        # hip positions for motion trail
        self._speed_history: deque = deque(maxlen=90) # speed for sparkline
        self._accel_burst: int = 0                   # frames to show accel burst
        self._fps_cache: float = 30.                 # cached from process_video

    # ── PROCESS VIDEO ─────────────────────────────────────────────────────────

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): raise FileNotFoundError(self.video_path)

        fps   = self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.
        self._fps_cache = fps
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(self.output_video_path, fourcc, fps, (W,H))
        print(f"[INFO] {total} frames @ {fps:.1f} fps  ({W}x{H})")

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            ts   = idx / fps
            bbox = self.tracker.update(frame)

            visible = False
            if bbox and bbox[2] > 20 and bbox[3] > 40:
                visible = True
                spd = self.frame_metrics[-1].speed if self.frame_metrics else 0.
                # Raw pose from contour estimator
                raw_kp = self.pose_est.estimate(frame, bbox, ts, spd)
                # Smooth every joint with Kalman filter
                kp     = self.smoother.smooth(raw_kp)

                pf = PoseFrame(idx, ts, bbox, kp)
                self.pose_frames.append(pf)
                if self.PIX_TO_M is None: self._calibrate(kp)

                fm = self._metrics(pf, idx, ts, fps)
                self.frame_metrics.append(fm)
                # Feed wow-factor state
                self._trail.append((int(kp.hip_center[0]), int(kp.hip_center[1])))
                self._speed_history.append(fm.speed)
                if abs(fm.acceleration) > 4.0:
                    self._accel_burst = 8   # hold burst effect for 8 frames
                elif self._accel_burst > 0:
                    self._accel_burst -= 1

                frame = self._draw_trail(frame)
                frame = self._annotate(frame, pf, fm, W, H)
                frame = self._draw_player_aura(frame, kp, fm)

            frame = self._hud(frame, idx, ts, total, visible)
            out.write(frame)
            idx += 1

        cap.release(); out.release()
        print(f"[INFO] Tracked in {len(self.pose_frames)}/{idx} frames")
        self._post_gait(fps)
        self._build_summary()
        print("[INFO] Done.")
        return self.summary

    # ── CALIBRATION ───────────────────────────────────────────────────────────

    def _calibrate(self, kp):
        leg = (dist2d(kp.left_hip, kp.left_ankle) +
               dist2d(kp.right_hip,kp.right_ankle)) / 2
        self.PIX_TO_M = 0.9/leg if leg>10 else 0.002

    # ── PER-FRAME METRICS ─────────────────────────────────────────────────────

    def _metrics(self, pf, idx, ts, fps) -> FrameMetrics:
        fm = FrameMetrics(frame_idx=idx, timestamp=ts)
        kp = pf.kp; sc = self.PIX_TO_M or 0.002

        fm.left_knee_angle  = angle_3pts(kp.left_hip,      kp.left_knee,  kp.left_ankle)
        fm.right_knee_angle = angle_3pts(kp.right_hip,     kp.right_knee, kp.right_ankle)
        fm.left_hip_angle   = angle_3pts(kp.left_shoulder, kp.left_hip,   kp.left_knee)
        fm.right_hip_angle  = angle_3pts(kp.right_shoulder,kp.right_hip,  kp.right_knee)

        dx = kp.shoulder_center[0]-kp.hip_center[0]
        dy = kp.shoulder_center[1]-kp.hip_center[1]
        fm.trunk_lean = math.degrees(math.atan2(abs(dx), abs(dy)+1e-9))

        # Knee valgus proxy (lateral knee deviation / hip width)
        hw = dist2d(kp.left_hip, kp.right_hip) + 1e-6
        fm.l_valgus = abs(kp.left_knee[0]  - kp.left_hip[0])  / hw
        fm.r_valgus = abs(kp.right_knee[0] - kp.right_hip[0]) / hw

        # Speed / acceleration
        if len(self.pose_frames) >= 2:
            prev = self.pose_frames[-2]
            dt   = ts - prev.timestamp + 1e-9
            dp   = dist2d(kp.hip_center, prev.kp.hip_center) * sc
            raw  = dp / dt
            self._spd_win.append(raw)
            fm.speed = float(np.mean(self._spd_win))
            fm.body_center_disp = dp
            if len(self.pose_frames) >= 3:
                p2  = self.pose_frames[-3]
                dp2 = dist2d(prev.kp.hip_center,p2.kp.hip_center)*sc
                dt2 = prev.timestamp-p2.timestamp+1e-9
                fm.acceleration = (raw - dp2/dt2)/dt

        # Direction change
        if len(self.pose_frames) >= 5:
            pos  = [p.kp.hip_center for p in list(self.pose_frames)[-5:]]
            vecs = [(pos[i+1][0]-pos[i][0],pos[i+1][1]-pos[i][1]) for i in range(4)]
            for i in range(len(vecs)-1):
                v1,v2=np.array(vecs[i]),np.array(vecs[i+1])
                n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
                if n1>2 and n2>2 and math.acos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))>math.radians(28):
                    fm.direction_change=True

        # Energy
        met = max(1.5, 3.5+fm.speed*2.3)
        fm.energy_expenditure = met*75

        # Joint stress (knee flexion)
        ks = sum((155-a)/155 for a in [fm.left_knee_angle,fm.right_knee_angle] if a<155)
        fm.joint_stress = min(1., ks/2)

        # Fatigue
        if len(self._spd_win) >= 10:
            s=list(self._spd_win)
            fm.fatigue_index=max(0.,min(1.,(np.mean(s[:5])-np.mean(s[-5:]))/(np.mean(s[:5])+1e-6)))

        # Composite risk score (0–100), same weighting as classmate but extended
        p_valgus = clamp01(((fm.l_valgus+fm.r_valgus)/2-0.02)/0.08)
        p_kasym  = clamp01(abs(fm.left_knee_angle-fm.right_knee_angle)/30.)
        p_accel  = clamp01(abs(fm.acceleration)/12)
        p_trunk  = clamp01(fm.trunk_lean/30)
        p_stress = fm.joint_stress
        raw_risk = (0.30*p_valgus + 0.25*p_stress +
                    0.20*p_kasym  + 0.15*p_accel  + 0.10*p_trunk)
        self._risk_win.append(raw_risk)
        fm.risk_score = float(np.mean(self._risk_win)) * 100.

        # Legacy scalar risks
        fm.fall_risk   = 0.  # filled in post_gait
        fm.injury_risk = raw_risk

        return fm

    # ── GAIT ANALYSIS ─────────────────────────────────────────────────────────

    def _post_gait(self, fps):
        if len(self.pose_frames) < 15: return
        sc   = self.PIX_TO_M or 0.002
        la_y = smooth_arr([p.kp.left_ankle[1]  for p in self.pose_frames])
        ra_y = smooth_arr([p.kp.right_ankle[1] for p in self.pose_frames])
        md   = max(5, int(fps*0.18))
        lp,_ = find_peaks(la_y, distance=md, prominence=2)
        rp,_ = find_peaks(ra_y, distance=md, prominence=2)

        pos   = [p.kp.hip_center for p in self.pose_frames]
        strl,stt,flt = [],[],[]
        for peaks in [lp,rp]:
            for i in range(1,len(peaks)):
                i0,i1=peaks[i-1],peaks[i]
                if i1>=len(pos): continue
                sl=dist2d(pos[i0],pos[i1])*sc
                if 0.15<sl<3.5: strl.append(sl)
                st=(i1-i0)/fps
                if 0.08<st<2.0: stt.append(st); flt.append(max(0.,st*0.35))

        n=min(len(lp),len(rp))-1
        if n>0:
            li=[(lp[i+1]-lp[i])/fps for i in range(n)]
            ri=[(rp[i+1]-rp[i])/fps for i in range(n)]
            m=min(len(li),len(ri))
            sym=float(np.mean([1-abs(l-r)/(l+r+1e-9) for l,r in zip(li[:m],ri[:m])]))*100
        else: sym=94.

        sv  = float(np.std(strl)/(np.mean(strl)+1e-9)*100) if len(strl)>2 else 3.5
        asl = float(np.mean(strl)) if strl else 1.35
        ast = float(np.mean(stt))  if stt  else 0.38
        aft = float(np.mean(flt))  if flt  else 0.13
        acad= 60./ast if ast>0 else 158.

        for fm in self.frame_metrics:
            fm.stride_length=asl; fm.step_time=ast; fm.flight_time=aft
            fm.cadence=acad; fm.gait_symmetry=sym; fm.stride_variability=sv

        for fm in self.frame_metrics:
            sym_r =max(0.,(100-fm.gait_symmetry)/100)
            var_r =min(1.,fm.stride_variability/25)
            lean_r=min(1.,fm.trunk_lean/40)
            fm.fall_risk=sym_r*0.4+var_r*0.3+lean_r*0.3
            acc_r=min(1.,abs(fm.acceleration)/12)
            fm.injury_risk=fm.joint_stress*0.5+acc_r*0.3+fm.fatigue_index*0.2

    # ── MOTION TRAIL ─────────────────────────────────────────────────────────

    def _draw_trail(self, frame):
        """Fading colour trail of the player's path."""
        pts = list(self._trail)
        if len(pts) < 2:
            return frame
        ov = frame.copy()
        for i in range(1, len(pts)):
            t     = i / len(pts)                       # 0=old, 1=newest
            alpha = t * 0.55
            col   = lerp_color((40,40,120), (0,220,255), t)
            thick = max(1, int(t * 4))
            cv2.line(ov, pts[i-1], pts[i], col, thick, cv2.LINE_AA)
        # Dot at oldest visible point
        cv2.circle(ov, pts[0], 3, (80,80,180), -1, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(ov, 0.55, frame, 0.45, 0)
        return frame

    # ── PLAYER AURA ───────────────────────────────────────────────────────────

    def _draw_player_aura(self, frame, kp, fm):
        """Speed-reactive glow ellipse around the player silhouette."""
        if fm.speed < 0.5:
            return frame
        hx, hy = int(kp.hip_center[0]), int(kp.hip_center[1])
        bx, by, bw, bh = self.pose_frames[-1].bbox
        rx = max(12, bw // 2 + 6)
        ry = max(20, bh // 2 + 10)
        intensity = clamp01(fm.speed / 8.)
        col = lerp_color((0,180,60), (0,60,255), intensity)

        # Accel burst: extra shockwave ring
        if self._accel_burst > 0:
            burst_r = int(rx * 1.6 + self._accel_burst * 3)
            burst_alpha = self._accel_burst / 8. * 0.4
            ov = frame.copy()
            cv2.ellipse(ov, (hx, hy), (burst_r, int(burst_r*1.4)),
                        0, 0, 360, (0,200,255), 3, cv2.LINE_AA)
            frame[:] = cv2.addWeighted(ov, burst_alpha, frame, 1-burst_alpha, 0)

        # Soft glow halo (2 layers)
        for expansion, a in [(14, 0.12), (6, 0.20)]:
            ov = frame.copy()
            cv2.ellipse(ov, (hx, hy), (rx+expansion, ry+expansion),
                        0, 0, 360, col, -1, cv2.LINE_AA)
            frame[:] = cv2.addWeighted(ov, a, frame, 1-a, 0)
        return frame

    # ── ANNOTATION ────────────────────────────────────────────────────────────

    def _annotate(self, frame, pf, fm, W, H):
        kp  = pf.kp
        rt  = clamp01(fm.risk_score / 100.)

        # Gradient skeleton with risk colouring
        render_skeleton(frame, kp, risk_tint=rt)

        hx, hy = int(kp.hip_center[0]), int(kp.hip_center[1])

        # ── Player ID badge (pill shape above head) ───────────────────────────
        head_y = int(kp.head[1]) - 28
        badge_txt = f"  #{self.player_id}  "
        (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        bx0 = hx - tw//2 - 4
        cv2.rectangle(frame, (bx0, head_y-20), (bx0+tw+8, head_y+4), (255,220,0), -1)
        cv2.rectangle(frame, (bx0, head_y-20), (bx0+tw+8, head_y+4), (0,0,0), 1)
        cv2.putText(frame, badge_txt, (bx0+4, head_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 2, cv2.LINE_AA)

        # ── Speed vector arrow ────────────────────────────────────────────────
        if fm.speed > 0.3 and len(self.pose_frames) >= 2:
            prev = self.pose_frames[-2]
            dx = kp.hip_center[0] - prev.kp.hip_center[0]
            dy = kp.hip_center[1] - prev.kp.hip_center[1]
            mag = math.hypot(dx, dy) + 1e-6
            ar  = int(min(fm.speed * 14, 90))
            ex, ey = int(hx + dx/mag*ar), int(hy + dy/mag*ar)
            sc2 = risk_color(fm.speed / 10. * 100)
            # Thick glow arrow
            cv2.arrowedLine(frame, (hx,hy), (ex,ey), sc2, 4, cv2.LINE_AA, tipLength=0.32)
            cv2.arrowedLine(frame, (hx,hy), (ex,ey), (255,255,255), 1, cv2.LINE_AA, tipLength=0.32)
            # Speed label with background pill
            spd_txt = f"{fm.speed:.1f} m/s"
            (stw, sth), _ = cv2.getTextSize(spd_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            sx, sy = ex+6, ey-4
            cv2.rectangle(frame, (sx-2, sy-sth-2), (sx+stw+4, sy+4), (0,0,0), -1)
            cv2.putText(frame, spd_txt, (sx, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, sc2, 2, cv2.LINE_AA)

        # ── Joint stress rings on knees ────────────────────────────────────────
        for kpt, ang, stress in [
            (kp.left_knee,  fm.left_knee_angle,  fm.joint_stress),
            (kp.right_knee, fm.right_knee_angle, fm.joint_stress),
        ]:
            kx, ky = int(kpt[0]), int(kpt[1])
            ac = (0,220,0) if ang>145 else (0,140,255) if ang>120 else (0,0,220)
            # Stress ring — pulses when under load
            if ang < 130:
                pulse = int(8 + 3 * math.sin(len(self.pose_frames) * 0.4))
                ov = frame.copy()
                cv2.circle(ov, (kx,ky), pulse, (0,0,255), 2, cv2.LINE_AA)
                frame[:] = cv2.addWeighted(ov, 0.6, frame, 0.4, 0)
            cv2.putText(frame, f"{ang:.0f}°", (kx+8, ky-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, ac, 1, cv2.LINE_AA)

        # ── Direction change flash ────────────────────────────────────────────
        if fm.direction_change:
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (W,H), (0,0,180), 6)
            frame[:] = cv2.addWeighted(ov, 0.55, frame, 0.45, 0)
            # Centred bold label
            label = "DIRECTION CHANGE"
            (lw,_),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
            cv2.rectangle(frame, (W//2-lw//2-10, H-52), (W//2+lw//2+10, H-16), (0,0,0), -1)
            cv2.putText(frame, label, (W//2-lw//2, H-24),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,100,255), 2, cv2.LINE_AA)

        # ── Mini risk gauge (arc) ─────────────────────────────────────────────
        gx, gy = W-70, H-80
        draw_risk_gauge(frame, gx, gy, 44, fm.risk_score)

        return frame

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_stat_bar(self, frame, x, y, w, h, value, max_val, col, label, val_fmt):
        """Horizontal animated stat bar: label | ████░░░ | value"""
        filled = int(w * clamp01(value / max(max_val, 1e-6)))
        # Background
        cv2.rectangle(frame, (x, y), (x+w, y+h), (35,35,35), -1)
        # Fill with gradient
        if filled > 0:
            for px in range(filled):
                t = px / max(w-1, 1)
                bc = lerp_color(col, lerp_color(col,(255,255,255),0.4), t)
                cv2.line(frame, (x+px, y+1), (x+px, y+h-1), bc, 1)
        # Border
        cv2.rectangle(frame, (x, y), (x+w, y+h), (60,60,60), 1)
        # Label left
        cv2.putText(frame, label, (x-2, y+h-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160,160,160), 1, cv2.LINE_AA)
        # Value right
        val_txt = val_fmt.format(value)
        (vw,_),_ = cv2.getTextSize(val_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, val_txt, (x+w+4, y+h-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    def _draw_sparkline(self, frame, x, y, w, h, values, col=(0,255,200)):
        """Mini speed graph — last N values drawn as a line chart."""
        vals = list(values)
        if len(vals) < 2:
            return
        cv2.rectangle(frame, (x, y), (x+w, y+h), (20,20,20), -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,50), 1)
        mx = max(max(vals), 0.1)
        pts = []
        for i, v in enumerate(vals):
            px = x + int(i / (len(vals)-1) * w)
            py = y + h - int(clamp01(v/mx) * (h-2)) - 1
            pts.append((px, py))
        for i in range(1, len(pts)):
            t = i / len(pts)
            lc = lerp_color((0,120,100), col, t)
            cv2.line(frame, pts[i-1], pts[i], lc, 2, cv2.LINE_AA)
        # Current value dot
        cv2.circle(frame, pts[-1], 3, (255,255,255), -1, cv2.LINE_AA)
        # Axis label
        cv2.putText(frame, f"{vals[-1]:.1f}", (x+w+3, y+h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)

    def _hud(self, frame, idx, ts, total, visible=True):
        H, W = frame.shape[:2]
        fps = self._fps_cache

        # ── Left panel (broadcast sidebar) ───────────────────────────────────
        PW = 230   # panel width
        PH = H
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (PW, PH), (8,8,12), -1)
        frame[:] = cv2.addWeighted(ov, 0.72, frame, 0.28, 0)

        # Gold accent line
        cv2.line(frame, (PW-1, 0), (PW-1, PH), (255,215,0), 2)

        # ── Header ────────────────────────────────────────────────────────────
        # Logo bar
        cv2.rectangle(frame, (0,0), (PW, 42), (20,20,28), -1)
        cv2.line(frame, (0,42), (PW,42), (255,215,0), 1)
        # Juventus B&W mini badge
        cv2.rectangle(frame, (6,6), (34,36), (255,255,255), -1)
        cv2.rectangle(frame, (20,6), (34,36), (0,0,0), -1)
        cv2.putText(frame, "JUV", (7,30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1)
        cv2.putText(frame, "ANALYTICS", (38,18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,215,0), 1, cv2.LINE_AA)
        cv2.putText(frame, "SPORTS SCIENCE", (38,34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160,160,160), 1, cv2.LINE_AA)

        # Tracker state pill
        t_state = self.tracker.state
        dot_col = {"pending":(0,200,255),"tracking":(0,220,0),"lost":(0,80,255)}.get(t_state,(150,150,150))
        state_txt = {"pending":"ACQUIRING","tracking":"LIVE","lost":"SEARCHING"}.get(t_state,"")
        cv2.circle(frame, (10,54), 5, dot_col, -1, cv2.LINE_AA)
        cv2.putText(frame, state_txt, (20,59),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot_col, 1, cv2.LINE_AA)
        # Timecode right-aligned
        tc = f"{int(ts//60):02d}:{ts%60:05.2f}"
        (tcw,_),_ = cv2.getTextSize(tc, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, tc, (PW-tcw-6, 59),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1, cv2.LINE_AA)

        # Player badge
        cv2.rectangle(frame, (0,66), (PW,92), (18,18,26), -1)
        cv2.putText(frame, f"PLAYER  #{self.player_id}", (8,84),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (255,215,0), 1, cv2.LINE_AA)

        if self.frame_metrics:
            fm = self.frame_metrics[-1]
            rc = risk_color(fm.risk_score)

            # ── Speed section ─────────────────────────────────────────────────
            y0 = 100
            cv2.putText(frame, "VELOCITY", (8, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120,120,120), 1, cv2.LINE_AA)
            cv2.line(frame, (0, y0+3), (PW, y0+3), (30,30,40), 1)

            # Big speed number
            spd_txt = f"{fm.speed:.1f}"
            cv2.putText(frame, spd_txt, (8, y0+38),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0,255,200), 2, cv2.LINE_AA)
            cv2.putText(frame, "m/s", (8 + int(len(spd_txt)*22), y0+38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80,200,160), 1, cv2.LINE_AA)

            # Sparkline
            self._draw_sparkline(frame, 8, y0+44, PW-20, 28,
                                 self._speed_history, (0,255,200))
            cv2.putText(frame, "3s SPEED HISTORY", (8, y0+84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80,80,80), 1, cv2.LINE_AA)

            # ── Stat bars ─────────────────────────────────────────────────────
            y1 = y0 + 96
            BAR_W = 80
            BAR_H = 7
            BX    = 68   # bar x start (after label)

            stats = [
                ("CADENCE",  fm.cadence,           220., (0,200,255),  "{:.0f} spm"),
                ("STRIDE",   fm.stride_length,      2.5,  (0,255,180),  "{:.2f} m"),
                ("ENERGY",   fm.energy_expenditure, 900., (0,180,255),  "{:.0f} kc"),
                ("GAIT SYM", fm.gait_symmetry,      100., (0,255,150),  "{:.0f}%"),
                ("TRUNK",    fm.trunk_lean,          30.,  (0,200,220),  "{:.1f}°"),
            ]
            for i, (lbl, val, mx, col, fmt) in enumerate(stats):
                sy = y1 + i * 22
                cv2.putText(frame, lbl, (6, sy+BAR_H),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110,110,110), 1, cv2.LINE_AA)
                self._draw_stat_bar(frame, BX, sy, BAR_W, BAR_H, val, mx, col, "", fmt)

            # ── Knee angles ───────────────────────────────────────────────────
            y2 = y1 + len(stats)*22 + 8
            cv2.putText(frame, "JOINT ANGLES", (8, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120,120,120), 1, cv2.LINE_AA)
            cv2.line(frame, (0, y2+3), (PW, y2+3), (30,30,40), 1)

            for ki, (side, ang, col) in enumerate([
                ("L KNEE", fm.left_knee_angle,  (255,180,0)),
                ("R KNEE", fm.right_knee_angle, (0,140,255)),
            ]):
                sy = y2 + 14 + ki*20
                arc_col = (0,220,0) if ang>145 else (0,140,255) if ang>120 else (0,0,220)
                cv2.putText(frame, side, (6, sy+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110,110,110), 1, cv2.LINE_AA)
                self._draw_stat_bar(frame, BX, sy, BAR_W, BAR_H, ang, 180., arc_col, "", "{:.0f}°")

            # ── Valgus ────────────────────────────────────────────────────────
            y3 = y2 + 14 + 2*20 + 6
            cv2.putText(frame, "VALGUS", (8, y3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120,120,120), 1, cv2.LINE_AA)
            cv2.line(frame, (0, y3+3), (PW, y3+3), (30,30,40), 1)
            for ki, (side, val) in enumerate([("L", fm.l_valgus), ("R", fm.r_valgus)]):
                sy = y3 + 14 + ki*20
                vc = (0,220,0) if val<0.1 else (0,140,255) if val<0.2 else (0,0,220)
                cv2.putText(frame, side, (6, sy+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110,110,110), 1, cv2.LINE_AA)
                self._draw_stat_bar(frame, BX, sy, BAR_W, BAR_H, val, 0.4, vc, "", "{:.2f}")

            # ── Risk section ──────────────────────────────────────────────────
            y4 = y3 + 14 + 2*20 + 8
            cv2.putText(frame, "INJURY RISK", (8, y4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120,120,120), 1, cv2.LINE_AA)
            cv2.line(frame, (0, y4+3), (PW, y4+3), (30,30,40), 1)

            # Big risk score
            risk_txt = f"{fm.risk_score:.0f}"
            cv2.putText(frame, risk_txt, (8, y4+38),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, rc, 2, cv2.LINE_AA)
            cv2.putText(frame, "/ 100", (8+int(len(risk_txt)*22), y4+38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1, cv2.LINE_AA)

            # Risk label pill
            risk_lbl = self._risk_label(fm.injury_risk)
            lbl_col  = (0,200,0) if risk_lbl=="Low" else (0,140,255) if risk_lbl=="Moderate" else (0,0,220)
            (rlw,_),_ = cv2.getTextSize(risk_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            rx0 = 8; ry0 = y4+44
            cv2.rectangle(frame, (rx0, ry0), (rx0+rlw+10, ry0+18), lbl_col, -1)
            cv2.putText(frame, risk_lbl, (rx0+5, ry0+13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1, cv2.LINE_AA)

            # Sub-risk bars
            for ki, (lbl, val, mx, col) in enumerate([
                ("FALL",    fm.fall_risk,    1., (0,200,255)),
                ("JOINT",   fm.joint_stress, 1., (0,140,255)),
                ("FATIGUE", fm.fatigue_index,1., (0,100,220)),
            ]):
                sy = y4 + 68 + ki*20
                cv2.putText(frame, lbl, (6, sy+8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100,100,100), 1, cv2.LINE_AA)
                self._draw_stat_bar(frame, BX, sy, BAR_W, 6, val, mx,
                                    risk_color(val*100), "", "{:.2f}")

        # ── Progress bar at very bottom of panel ──────────────────────────────
        prog = clamp01(ts / max(total/fps, 1e-6))
        pbar_y = H - 18
        cv2.rectangle(frame, (0, pbar_y), (PW, H), (15,15,20), -1)
        if prog > 0:
            cv2.rectangle(frame, (0, pbar_y+2), (int(PW*prog), H-2), (255,215,0), -1)
        cv2.putText(frame, f"{prog*100:.0f}%", (PW//2-10, H-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,0,0) if prog>0.3 else (120,120,120),
                    1, cv2.LINE_AA)

        # ── Risk bar along bottom of whole frame ──────────────────────────────
        if self.frame_metrics:
            rc2 = risk_color(self.frame_metrics[-1].risk_score)
            bw  = int((W-PW) * clamp01(self.frame_metrics[-1].risk_score/100.))
            cv2.rectangle(frame, (PW, H-8), (W, H), (20,20,20), -1)
            if bw > 0:
                cv2.rectangle(frame, (PW, H-8), (PW+bw, H), rc2, -1)

        # ── Out-of-frame banner ───────────────────────────────────────────────
        if not visible and t_state == "lost":
            bov = frame.copy()
            cv2.rectangle(bov, (PW, H//2-35), (W, H//2+35), (0,0,0), -1)
            frame[:] = cv2.addWeighted(bov, 0.70, frame, 0.30, 0)
            cv2.putText(frame, f"PLAYER #{self.player_id} — OUT OF FRAME",
                        (PW + 20, H//2+8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,140,255), 2, cv2.LINE_AA)

        return frame

    # ── SUMMARY ───────────────────────────────────────────────────────────────

    def _build_summary(self):
        if not self.frame_metrics: return
        fms=self.frame_metrics; s=self.summary
        sc=self.PIX_TO_M or 0.002

        s.total_frames=len(fms); s.duration_seconds=fms[-1].timestamp
        spds=np.array([f.speed for f in fms])
        s.avg_speed=float(np.mean(spds)); s.max_speed=float(np.max(spds))

        def anz(a): v=[getattr(f,a) for f in fms if getattr(f,a)>0]; return float(np.mean(v)) if v else 0.
        s.avg_stride_length=anz("stride_length"); s.avg_step_time=anz("step_time")
        s.avg_cadence=anz("cadence"); s.avg_flight_time=anz("flight_time")
        s.estimated_energy_kcal_hr=float(np.mean([f.energy_expenditure for f in fms]))
        s.gait_symmetry_pct=float(np.mean([f.gait_symmetry for f in fms]))
        s.stride_variability_pct=float(np.mean([f.stride_variability for f in fms]))
        dc=sum(1 for f in fms if f.direction_change)
        s.direction_change_freq=dc/max(s.duration_seconds/60,1e-6)
        s.peak_risk_score=float(np.max([f.risk_score for f in fms]))

        if len(self.pose_frames)>=2:
            s.total_distance_m=sum(
                dist2d(self.pose_frames[i].kp.hip_center,
                       self.pose_frames[i-1].kp.hip_center)*sc
                for i in range(1,len(self.pose_frames)))

        def rl(a): return self._risk_label(float(np.mean([getattr(f,a) for f in fms])))
        s.fall_risk_label=rl("fall_risk"); s.injury_risk_label=rl("injury_risk")
        s.body_stress_label=rl("joint_stress"); s.fatigue_label=rl("fatigue_index")
        ai=float(np.mean([f.injury_risk for f in fms]))
        s.injury_risk_detail=("high knee load" if ai>0.5
                              else "moderate joint stress" if ai>0.3
                              else "within normal range")

    @staticmethod
    def _risk_label(v): return "Low" if v<0.25 else "Moderate" if v<0.55 else "High"

    # ── EXPORT ────────────────────────────────────────────────────────────────

    def export_json(self, path):
        with open(path,"w") as f:
            json.dump({"player_summary":asdict(self.summary),
                       "frame_metrics":[asdict(m) for m in self.frame_metrics]},f,indent=2)
        print(f"[EXPORT] JSON → {path}")

    def export_csv(self, path):
        pd.DataFrame([asdict(m) for m in self.frame_metrics]).to_csv(path,index=False)
        print(f"[EXPORT] CSV  → {path}")

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(m) for m in self.frame_metrics])

    def print_report(self):
        s=self.summary
        print("\n"+"═"*54)
        print(f"  JUVENTUS SPORTS ANALYTICS  —  Player #{s.player_id}")
        print("═"*54)
        print(f"  Duration        : {s.duration_seconds:.1f}s  ({s.total_frames} frames)")
        print(f"  Total Distance  : {s.total_distance_m:.1f} m")
        print()
        print("  ── Player Metrics ──────────────────────────────")
        print(f"  Speed (avg / max)        : {s.avg_speed:.2f} / {s.max_speed:.2f} m/s")
        print(f"  Stride Length            : {s.avg_stride_length:.2f} m")
        print(f"  Step Time                : {s.avg_step_time:.2f} s")
        print(f"  Cadence                  : {s.avg_cadence:.0f} steps/min")
        print(f"  Flight Time              : {s.avg_flight_time:.3f} s")
        print(f"  Direction Change Freq    : {s.direction_change_freq:.1f} / min")
        print(f"  Est. Energy Expenditure  : {s.estimated_energy_kcal_hr:.0f} kcal/hour")
        print()
        print("  ── Risk Indicators ─────────────────────────────")
        print(f"  Peak Risk Score    : {s.peak_risk_score:.0f}/100")
        print(f"  Fall Risk          : {s.fall_risk_label}")
        print(f"  Injury Risk        : {s.injury_risk_label} ({s.injury_risk_detail})")
        print(f"  Gait Symmetry      : {s.gait_symmetry_pct:.1f}%")
        print(f"  Body Stress Level  : {s.body_stress_label}")
        print(f"  Fatigue            : {s.fatigue_label}")
        print("═"*54+"\n")