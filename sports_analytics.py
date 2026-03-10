"""
Juventus Sports Analytics System  v4
======================================
Tracking architecture upgrade — see architecture diagram at bottom of imports.

Install on your Windows machine:
    pip install ultralytics          # YOLO11 pose model
    pip install opencv-python numpy pandas scipy
"""

import cv2
import numpy as np
import pandas as pd
import json
import math
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

try:
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("[WARNING] ultralytics not found — pip install ultralytics")
    print("          Falling back to MOG2 blob detection.")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

JOINT_NAMES = [
    "head","neck","left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_foot","right_foot","hip_center","shoulder_center",
]

@dataclass
class PoseKeypoints:
    head:(float,float)=(0.,0.); neck:(float,float)=(0.,0.)
    left_shoulder:(float,float)=(0.,0.); right_shoulder:(float,float)=(0.,0.)
    left_elbow:(float,float)=(0.,0.);    right_elbow:(float,float)=(0.,0.)
    left_wrist:(float,float)=(0.,0.);    right_wrist:(float,float)=(0.,0.)
    left_hip:(float,float)=(0.,0.);      right_hip:(float,float)=(0.,0.)
    left_knee:(float,float)=(0.,0.);     right_knee:(float,float)=(0.,0.)
    left_ankle:(float,float)=(0.,0.);    right_ankle:(float,float)=(0.,0.)
    left_foot:(float,float)=(0.,0.);     right_foot:(float,float)=(0.,0.)
    hip_center:(float,float)=(0.,0.);    shoulder_center:(float,float)=(0.,0.)

@dataclass
class PoseFrame:
    frame_idx:int; timestamp:float; bbox:Tuple[int,int,int,int]; kp:PoseKeypoints

@dataclass
class FrameMetrics:
    frame_idx:int=0; timestamp:float=0.; speed:float=0.; acceleration:float=0.
    stride_length:float=0.; step_time:float=0.; cadence:float=0.; flight_time:float=0.
    left_knee_angle:float=0.; right_knee_angle:float=0.
    left_hip_angle:float=0.;  right_hip_angle:float=0.
    trunk_lean:float=0.; direction_change:bool=False; energy_expenditure:float=0.
    gait_symmetry:float=100.; stride_variability:float=0.
    fall_risk:float=0.; injury_risk:float=0.; joint_stress:float=0.
    fatigue_index:float=0.; body_center_disp:float=0.
    l_valgus:float=0.; r_valgus:float=0.; risk_score:float=0.

@dataclass
class PlayerSummary:
    player_id:int=1; total_frames:int=0; duration_seconds:float=0.
    avg_speed:float=0.; max_speed:float=0.; avg_stride_length:float=0.
    avg_step_time:float=0.; avg_cadence:float=0.; avg_flight_time:float=0.
    direction_change_freq:float=0.; estimated_energy_kcal_hr:float=0.
    gait_symmetry_pct:float=0.; stride_variability_pct:float=0.
    total_distance_m:float=0.; peak_risk_score:float=0.
    fall_risk_label:str="Low"; injury_risk_label:str="Low"
    injury_risk_detail:str=""; body_stress_label:str="Low"; fatigue_label:str="Low"


# ══════════════════════════════════════════════════════════════════════════════
#  MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def angle_3pts(a,b,c)->float:
    ba=np.array(a)-np.array(b); bc=np.array(c)-np.array(b)
    n=np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/n,-1,1))))

def dist2d(p1,p2)->float: return math.hypot(p1[0]-p2[0],p1[1]-p2[1])

def smooth_arr(arr,w=5):
    a=np.array(arr,dtype=float)
    if HAS_SCIPY: return uniform_filter1d(a,size=w)
    return np.convolve(a,np.ones(w)/w,mode='same')

def clamp01(x): return float(np.clip(x,0.,1.))

def lerp_color(c1,c2,t):
    t=clamp01(t); return tuple(int(c1[i]*(1-t)+c2[i]*t) for i in range(3))

def risk_color(s):
    t=clamp01(s/100.)
    return lerp_color((0,200,0),(0,200,255),t*2) if t<0.5 else lerp_color((0,200,255),(0,0,230),(t-.5)*2)

def bbox_iou(a,b)->float:
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ix=max(0,min(ax+aw,bx+bw)-max(ax,bx)); iy=max(0,min(ay+ah,by+bh)-max(ay,by))
    inter=ix*iy; return inter/(aw*ah+bw*bh-inter+1e-6)

def bbox_centre(bbox): x,y,w,h=bbox; return (x+w/2.,y+h/2.)

def crop_hist(frame,bbox):
    bx,by,bw,bh=[int(v) for v in bbox]; H,W=frame.shape[:2]
    bx,by=max(0,bx),max(0,by); bw,bh=min(bw,W-bx),min(bh,H-by)
    if bw<5 or bh<5: return None
    hsv=cv2.cvtColor(frame[by:by+bh,bx:bx+bw],cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[18,16],[0,180,0,256])
    cv2.normalize(hist,hist); return hist

def hist_sim(h1,h2)->float:
    if h1 is None or h2 is None: return 0.
    return float(cv2.compareHist(h1,h2,cv2.HISTCMP_CORREL))


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN TRACK  — per-player state: [cx,cy,w,h, vx,vy,vw,vh]
# ══════════════════════════════════════════════════════════════════════════════

class KalmanTrack:
    _next_id=1
    F=np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],
                [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],dtype=float)
    H=np.eye(4,8)

    def __init__(self,bbox,frame,conf=1.0):
        self.id=KalmanTrack._next_id; KalmanTrack._next_id+=1
        cx,cy=bbox_centre(bbox); w,h=bbox[2],bbox[3]
        self.x=np.array([cx,cy,w,h,0.,0.,0.,0.],dtype=float)
        self.P=np.diag([10.,10.,10.,10.,100.,100.,10.,10.])
        self.Q=np.diag([1.,1.,1.,1.,.5,.5,.2,.2])
        self.R=np.diag([4.,4.,10.,10.])
        self.conf=conf; self.hit_streak=1; self.missed=0; self.age=1
        self.ref_hist=crop_hist(frame,bbox); self.last_bbox=bbox
        self.trajectory=deque(maxlen=30); self.trajectory.append(bbox_centre(bbox))
        self._yolo_kp=None

    def predict(self):
        self.x=self.F@self.x; self.P=self.F@self.P@self.F.T+self.Q
        self.x[2]=max(1.,self.x[2]); self.x[3]=max(1.,self.x[3])
        self.age+=1; self.missed+=1; return self.get_bbox()

    def update(self,bbox,frame,conf=1.0):
        cx,cy=bbox_centre(bbox); z=np.array([cx,cy,bbox[2],bbox[3]],dtype=float)
        S=self.H@self.P@self.H.T+self.R; K=self.P@self.H.T@np.linalg.inv(S)
        self.x=self.x+K@(z-self.H@self.x); self.P=(np.eye(8)-K@self.H)@self.P
        self.conf=conf; self.hit_streak+=1; self.missed=0; self.last_bbox=bbox
        self.trajectory.append(bbox_centre(bbox))
        nh=crop_hist(frame,bbox)
        if nh is not None and self.ref_hist is not None:
            self.ref_hist=(0.92*self.ref_hist+0.08*nh).astype(np.float32)
            cv2.normalize(self.ref_hist,self.ref_hist)
        elif nh is not None: self.ref_hist=nh

    def get_bbox(self)->Tuple[int,int,int,int]:
        cx,cy,w,h=self.x[:4]; return (int(cx-w/2),int(cy-h/2),int(w),int(h))

    def reactivate(self,bbox,frame):
        cx,cy=bbox_centre(bbox); self.x[:4]=[cx,cy,bbox[2],bbox[3]]
        self.missed=0; self.hit_streak=1; self.last_bbox=bbox
        nh=crop_hist(frame,bbox)
        if nh is not None: self.ref_hist=nh


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION LAYER  — YOLO first, MOG2 blob fallback
# ══════════════════════════════════════════════════════════════════════════════

class DetectionLayer:
    def __init__(self,model_size="m"):
        self._yolo=None
        self._bg=cv2.createBackgroundSubtractorMOG2(history=400,varThreshold=40,detectShadows=False)
        self._mode="blob"
        if HAS_YOLO:
            try:
                mn=f"yolo11{model_size}-pose.pt"
                print(f"[DETECT] Loading {mn} …")
                self._yolo=_YOLO(mn)
                self._mode="yolo"
                print("[DETECT] YOLO pose model loaded.")
            except Exception as e:
                print(f"[DETECT] YOLO failed ({e}) — blob fallback.")

    @property
    def mode(self): return self._mode

    def detect(self,frame)->List[dict]:
        return self._yolo_detect(frame) if self._mode=="yolo" else self._blob_detect(frame)

    def _yolo_detect(self,frame)->List[dict]:
        res=self._yolo(frame,verbose=False,conf=0.25)[0]; dets=[]
        if res.boxes is None or len(res.boxes)==0: return dets
        for i,box in enumerate(res.boxes):
            x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
            bw,bh=x2-x1,y2-y1
            if bh<bw*0.8: continue
            bbox=(int(x1),int(y1),int(bw),int(bh))
            conf=float(box.conf[0].cpu()); kp=None
            if res.keypoints is not None and i<len(res.keypoints.xy):
                kpxy=res.keypoints.xy[i].cpu().numpy()
                kpc=res.keypoints.conf[i].cpu().numpy()
                kpxy[kpc<0.3]=0.; kp=kpxy
            dets.append({'bbox':bbox,'conf':conf,'kp':kp})
        return dets

    def _blob_detect(self,frame)->List[dict]:
        mask=self._bg.apply(frame)
        k7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        k3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k7,iterations=2)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k3,iterations=1)
        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cands=[]
        for cnt in cnts:
            area=cv2.contourArea(cnt)
            if not (2000<=area<=90000): continue
            bx,by,bw,bh=cv2.boundingRect(cnt)
            if not (1.3<=bh/(bw+1e-6)<=5.0): continue
            fill=area/(bw*bh+1e-6)
            if fill<0.25: continue
            cands.append({'bbox':(bx,by,bw,bh),'conf':fill,'kp':None,'area':area})
        cands.sort(key=lambda c:c['area'],reverse=True)
        kept,sup=[], set()
        for i,ci in enumerate(cands):
            if i in sup: continue
            kept.append(ci)
            for j,cj in enumerate(cands):
                if j<=i or j in sup: continue
                if bbox_iou(ci['bbox'],cj['bbox'])>0.40: sup.add(j)
        return kept


# ══════════════════════════════════════════════════════════════════════════════
#  SCENE-CHANGE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class SceneChangeDetector:
    def __init__(self,threshold=0.45): self._prev=None; self._thr=threshold
    def is_cut(self,frame)->bool:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        hist=cv2.calcHist([gray],[0],None,[64],[0,256]); cv2.normalize(hist,hist)
        if self._prev is None: self._prev=hist; return False
        score=float(cv2.compareHist(self._prev,hist,cv2.HISTCMP_CORREL))
        self._prev=hist; return score<self._thr


# ══════════════════════════════════════════════════════════════════════════════
#  BYTETRACKER  — two-stage association, handles occlusion via Kalman + TTL
# ══════════════════════════════════════════════════════════════════════════════

class ByteTracker:
    HIGH_THRESH=0.50; LOW_THRESH=0.20
    IOU_HIGH=0.30;    IOU_LOW=0.15;   IOU_LOST=0.20
    MIN_HITS=2;       LOST_TTL=60

    def __init__(self):
        self.active_tracks:List[KalmanTrack]=[]
        self.lost_tracks:List[KalmanTrack]=[]

    def update(self,detections:List[dict],frame)->List[KalmanTrack]:
        for t in self.active_tracks+self.lost_tracks: t.predict()
        high=[d for d in detections if d['conf']>=self.HIGH_THRESH]
        low =[d for d in detections if self.LOW_THRESH<=d['conf']<self.HIGH_THRESH]

        unm_t, unm_h = self._associate(self.active_tracks, high, frame, self.IOU_HIGH)
        still_unm, _ = self._associate(unm_t, low, frame, self.IOU_LOW)
        self._associate(self.lost_tracks, low, frame, self.IOU_LOST, reactivate=True)

        for t in still_unm:
            t.hit_streak=0
            if t not in self.lost_tracks: self.lost_tracks.append(t)
            if t in self.active_tracks: self.active_tracks.remove(t)

        for d in unm_h:
            self.active_tracks.append(KalmanTrack(d['bbox'],frame,d['conf']))

        self.lost_tracks=[t for t in self.lost_tracks if t.missed<=self.LOST_TTL]
        return [t for t in self.active_tracks if t.hit_streak>=self.MIN_HITS]

    def _associate(self,tracks,dets,frame,iou_thr,reactivate=False):
        if not tracks or not dets: return list(tracks),list(dets)
        cost=np.zeros((len(tracks),len(dets)),dtype=float)
        for ti,t in enumerate(tracks):
            tb=t.get_bbox(); th=t.ref_hist
            for di,d in enumerate(dets):
                iou=bbox_iou(tb,d['bbox']); hs=hist_sim(th,crop_hist(frame,d['bbox']))
                cost[ti,di]=1.0-(iou*0.60+hs*0.40)
        mt,md=set(),set()
        while True:
            avail=[(ti,di) for ti in range(len(tracks)) for di in range(len(dets))
                   if ti not in mt and di not in md]
            if not avail: break
            ti,di=min(avail,key=lambda p:cost[p[0],p[1]])
            if cost[ti,di]>=1.0-iou_thr: break
            mt.add(ti); md.add(di)
            t=tracks[ti]; d=dets[di]
            t.update(d['bbox'],frame,d['conf'])
            if reactivate:
                t.reactivate(d['bbox'],frame)
                if t in self.lost_tracks: self.lost_tracks.remove(t)
                if t not in self.active_tracks: self.active_tracks.append(t)
            if d.get('kp') is not None: t._yolo_kp=d['kp']
        return ([tracks[i] for i in range(len(tracks)) if i not in mt],
                [dets[i]   for i in range(len(dets))   if i not in md])

    def reset(self):
        for t in self.active_tracks+self.lost_tracks: t.x[4:]=0.
        print("[TRACKER] Scene cut — velocity reset.")


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET LOCK  — single-player selector on top of ByteTracker
#  Handles: occlusion, camera cuts, player overlap / identity swap
# ══════════════════════════════════════════════════════════════════════════════

class TargetLock:
    def __init__(self,seed_bbox,seed_hist,seed_frame_idx):
        self._seed_bbox=seed_bbox; self._ref_hist=seed_hist
        self._seed_fi=seed_frame_idx; self._target_id=None
        self._last_bbox=None; self._smooth_box=None; self._alpha=0.35
        self._state="searching"; self._lost_frames=0; self._fi=0
        self.bt=ByteTracker(); self.scene=SceneChangeDetector()

    @property
    def state(self): return self._state
    @property
    def lost_count(self): return self._lost_frames

    def update(self,frame)->Optional[Tuple]:
        # Scene cut
        if self.scene.is_cut(frame) and self._fi>10:
            print(f"[LOCK] Scene cut @ frame {self._fi}")
            self.bt.reset(); self._target_id=None; self._state="searching"

        dets=_detection_layer.detect(frame)
        tracks=self.bt.update(dets,frame)
        self._fi+=1

        # Initial lock
        if self._target_id is None:
            if self._fi>=self._seed_fi:
                self._target_id=self._choose(tracks,frame)
                if self._target_id is not None:
                    print(f"[LOCK] Locked id={self._target_id} @ frame {self._fi}")
                    self._state="tracking"
            return None

        target=next((t for t in tracks if t.id==self._target_id),None)
        if target is not None:
            target=self._resolve_overlap(target,tracks)

        if target is None:
            self._lost_frames+=1; self._state="lost"
            target=self._reacquire(tracks,strict=self._lost_frames<=5)
            if target is None:
                target=self._reacquire(self.bt.lost_tracks,strict=False)
        else:
            self._lost_frames=0; self._state="tracking"

        if target is None: return None
        self._target_id=target.id; self._last_bbox=target.get_bbox()
        return self._emit(self._last_bbox)

    def _choose(self,tracks,frame)->Optional[int]:
        if not tracks: return None
        best,bid=-1.,None
        for t in tracks:
            iou=bbox_iou(t.get_bbox(),self._seed_bbox)
            hs=hist_sim(t.ref_hist,self._ref_hist)
            sw,sh=self._seed_bbox[2],self._seed_bbox[3]
            tw,th=t.get_bbox()[2],t.get_bbox()[3]
            ss=min(sw*sh,tw*th)/(max(sw*sh,tw*th)+1e-6)
            sc=iou*0.45+hs*0.40+ss*0.15
            if sc>best: best,bid=sc,t.id
        return bid

    def _reacquire(self,tracks,strict=True)->Optional[KalmanTrack]:
        if not tracks: return None
        thr=0.35 if strict else 0.18; best,bt=-1.,None
        for t in tracks:
            hs=hist_sim(t.ref_hist,self._ref_hist)
            if hs<thr: continue
            if self._last_bbox is not None:
                lw,lh=self._last_bbox[2],self._last_bbox[3]
                tw,th=t.get_bbox()[2],t.get_bbox()[3]
                ss=min(lw*lh,tw*th)/(max(lw*lh,tw*th)+1e-6)
                if ss<0.25: continue
                sc=hs*0.65+ss*0.35
            else: sc=hs
            if sc>best: best,bt=sc,t
        if bt is not None:
            print(f"[LOCK] Re-acquired id={bt.id} (score={best:.2f}) @ frame {self._fi}")
            self._target_id=bt.id; self._state="tracking"
        return bt

    def _resolve_overlap(self,target,tracks)->KalmanTrack:
        tb=target.get_bbox()
        for other in tracks:
            if other.id==target.id: continue
            if bbox_iou(tb,other.get_bbox())>0.55:
                ts=hist_sim(target.ref_hist,self._ref_hist)
                os=hist_sim(other.ref_hist,self._ref_hist)
                if os>ts+0.12:
                    print(f"[LOCK] Overlap swap {target.id}->{other.id} @ frame {self._fi}")
                    self._target_id=other.id; return other
        return target

    def _emit(self,bbox)->Tuple:
        arr=np.array(bbox,dtype=float)
        if self._smooth_box is None: self._smooth_box=arr
        else: self._smooth_box=self._alpha*arr+(1-self._alpha)*self._smooth_box
        return tuple(int(v) for v in self._smooth_box)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL DETECTION SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_detection_layer:Optional[DetectionLayer]=None

def _get_detection_layer(model_size="m")->DetectionLayer:
    global _detection_layer
    if _detection_layer is None: _detection_layer=DetectionLayer(model_size)
    return _detection_layer


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PLAYER PICKER
# ══════════════════════════════════════════════════════════════════════════════

def pick_player_interactive(video_path:str)->Optional[dict]:
    det=_get_detection_layer()
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise FileNotFoundError(video_path)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WARMUP=min(90,total//3); cands=[]
    for fi in range(WARMUP):
        ret,frame=cap.read()
        if not ret: break
        dets=det.detect(frame)
        if dets: cands.append((frame.copy(),dets,fi))
    cap.release()
    if not cands:
        print("[PICKER] No detections — auto-select.")
        return select_primary_player(video_path)
    best_frame,best_dets,best_fi=max(cands,key=lambda c:len(c[1]))
    display=cv2.addWeighted(best_frame.copy(),0.65,np.zeros_like(best_frame),0.35,0)
    COLORS=[(0,255,180),(0,140,255),(255,215,0),(0,200,255),(180,0,255),(0,255,80),(255,80,80),(80,255,255)]
    blobs=[d['bbox'] for d in best_dets]
    for i,(bx,by,bw,bh) in enumerate(blobs):
        col=COLORS[i%len(COLORS)]
        cv2.rectangle(display,(bx,by),(bx+bw,by+bh),col,3,cv2.LINE_AA)
        lbl=str(i+1); lw,_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)[0]
        bxb=bx+bw//2-lw//2-6; byb=max(0,by-34)
        cv2.rectangle(display,(bxb,byb),(bxb+lw+12,byb+28),col,-1)
        cv2.putText(display,lbl,(bxb+6,byb+22),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2,cv2.LINE_AA)
    BH=52; banner=np.full((BH,W,3),15,np.uint8)
    mode_str=det.mode.upper()
    cv2.putText(banner,f"[{mode_str}] CLICK player to track  |  ESC=auto",(W//2-260,34),
                cv2.FONT_HERSHEY_SIMPLEX,0.68,(255,215,0),1,cv2.LINE_AA)
    display=np.vstack([banner,display])
    chosen=[None]
    def on_click(ev,cx,cy,fl,p):
        if ev!=cv2.EVENT_LBUTTONDOWN: return
        ay=cy-BH
        if ay<0: return
        for b in blobs:
            bx,by,bw,bh=b
            if bx<=cx<=bx+bw and by<=ay<=by+bh: chosen[0]=b; return
        chosen[0]=min(blobs,key=lambda b:math.hypot(cx-(b[0]+b[2]/2),ay-(b[1]+b[3]/2)))
    cv2.namedWindow("Select Player",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Player",min(W,1280),min(H+BH,800))
    cv2.setMouseCallback("Select Player",on_click)
    print(f"\n[PICKER] {len(blobs)} player(s) [{mode_str}].  Click to select, ESC=auto.\n")
    while True:
        cv2.imshow("Select Player",display)
        if chosen[0] is not None or (cv2.waitKey(20)&0xFF)==27: break
    cv2.destroyAllWindows()
    if chosen[0] is None:
        print("[PICKER] ESC — auto-select."); return select_primary_player(video_path)
    blob=chosen[0]; bx,by,bw,bh=blob
    print(f"[PICKER] Selected bbox={blob}")
    return {'hist':crop_hist(best_frame,blob),'size':(float(bw),float(bh)),
            'seed_bbox':blob,'seed_frame':best_fi}


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO PRE-SCAN
# ══════════════════════════════════════════════════════════════════════════════

def select_primary_player(video_path:str,sample_step:int=6)->Optional[dict]:
    det=_get_detection_layer()
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracks:List[dict]=[]; MAX_GAP=max(sample_step*5,30); fi=0
    print(f"[PRE-SCAN] {total} frames (step={sample_step}) …")
    while True:
        ret,frame=cap.read()
        if not ret: break
        if fi%sample_step==0:
            for d in det.detect(frame):
                blob=d['bbox']; bx,by,bw,bh=blob; matched=False
                for tr in tracks:
                    if fi-tr["lf"]>MAX_GAP: continue
                    iou=bbox_iou(blob,tr["lb"])
                    rw,rh=tr["ms"]
                    ss=min(bw*bh,rw*rh)/(max(bw*bh,rw*rh)+1e-6)
                    if iou*0.7+ss*0.3>0.15 and (iou>0.10 or ss>0.55):
                        h=crop_hist(frame,blob); tr["n"]+=1
                        if h is not None: tr["hs"].append(h)
                        n=tr["n"]; pw,ph=tr["ms"]
                        tr["ms"]=((pw*(n-1)+bw)/n,(ph*(n-1)+bh)/n)
                        tr["lb"]=blob; tr["lf"]=fi; matched=True; break
                if not matched:
                    h=crop_hist(frame,blob)
                    tracks.append({"n":1,"hs":[h] if h is not None else [],"ms":(float(bw),float(bh)),
                                   "lb":blob,"lf":fi,"sb":blob,"sf":fi})
        fi+=1
    cap.release()
    if not tracks: return None
    best=max(tracks,key=lambda t:t["n"])
    print(f"[PRE-SCAN] Best: {best['n']} hits, seed={best['sb']}")
    mh=None
    if best["hs"]:
        stacked=np.mean(best["hs"],axis=0).astype(np.float32); cv2.normalize(stacked,stacked); mh=stacked
    return {'hist':mh,'size':best["ms"],'seed_bbox':best["sb"],'seed_frame':best["sf"]}


# ══════════════════════════════════════════════════════════════════════════════
#  HYBRID POSE ESTIMATOR  — YOLO keypoints + geometric fallback
# ══════════════════════════════════════════════════════════════════════════════

_COCO={"nose":0,"left_shoulder":5,"right_shoulder":6,"left_elbow":7,"right_elbow":8,
       "left_wrist":9,"right_wrist":10,"left_hip":11,"right_hip":12,
       "left_knee":13,"right_knee":14,"left_ankle":15,"right_ankle":16}

class HybridPoseEstimator:
    _VP=dict(head=0.04,neck=0.11,shoulder=0.20,elbow=0.34,wrist=0.46,
             hip=0.54,knee=0.73,ankle=0.91,foot=0.99)

    def __init__(self): self._prev_cx=None; self._dh=deque(maxlen=8)

    def estimate(self,frame,bbox,ts,spd=0.,yolo_kp=None)->PoseKeypoints:
        x,y,w,h=bbox; cx=x+w/2.
        disp=abs(cx-self._prev_cx) if self._prev_cx is not None else 0.
        self._prev_cx=cx; self._dh.append(disp)
        ds=sum(self._dh); phase=(ds/max(w*0.18,4.))*math.pi
        swing=clamp01(spd/9.); arm_sw=swing*0.10*w; leg_sw=swing*0.08*w; k_lift=swing*0.08*h
        cw=self._cwidths(frame,bbox); sh,hh=self._bwidths(cw,w,h)
        def vy(f): return y+f*h
        kp=PoseKeypoints()
        kp.head=(cx,vy(self._VP["head"])); kp.neck=(cx,vy(self._VP["neck"]))
        ls=(cx-sh,vy(self._VP["shoulder"])); rs=(cx+sh,vy(self._VP["shoulder"]))
        kp.left_shoulder=ls; kp.right_shoulder=rs
        kp.shoulder_center=((ls[0]+rs[0])/2.,(ls[1]+rs[1])/2.)
        aoff=arm_sw*math.sin(phase)
        le=(ls[0]-aoff,vy(self._VP["elbow"])); re=(rs[0]+aoff,vy(self._VP["elbow"]))
        kp.left_elbow=le; kp.right_elbow=re
        kp.left_wrist=(le[0]-aoff*.55,vy(self._VP["wrist"]))
        kp.right_wrist=(re[0]+aoff*.55,vy(self._VP["wrist"]))
        lh=(cx-hh,vy(self._VP["hip"])); rh=(cx+hh,vy(self._VP["hip"]))
        kp.left_hip=lh; kp.right_hip=rh
        kp.hip_center=((lh[0]+rh[0])/2.,(lh[1]+rh[1])/2.)
        loff=leg_sw*math.sin(phase); roff=-loff
        ll=k_lift*max(0.,math.sin(phase)); rl=k_lift*max(0.,-math.sin(phase))
        kp.left_knee=(lh[0]+loff,vy(self._VP["knee"])-ll)
        kp.right_knee=(rh[0]+roff,vy(self._VP["knee"])-rl)
        kp.left_ankle=(lh[0]+loff*.45,vy(self._VP["ankle"])-ll*.5)
        kp.right_ankle=(rh[0]+roff*.45,vy(self._VP["ankle"])-rl*.5)
        kp.left_foot=(kp.left_ankle[0]+w*.07,vy(self._VP["foot"]))
        kp.right_foot=(kp.right_ankle[0]+w*.07,vy(self._VP["foot"]))
        # Overlay YOLO where available
        if yolo_kp is not None and len(yolo_kp)==17:
            def g(nm):
                i=_COCO.get(nm)
                if i is None: return None
                pt=yolo_kp[i]
                return (float(pt[0]),float(pt[1])) if (pt[0]>1 or pt[1]>1) else None
            def gxy(nm,df): p=g(nm); return p if p is not None else df
            kp.left_shoulder=gxy("left_shoulder",kp.left_shoulder)
            kp.right_shoulder=gxy("right_shoulder",kp.right_shoulder)
            kp.left_elbow=gxy("left_elbow",kp.left_elbow)
            kp.right_elbow=gxy("right_elbow",kp.right_elbow)
            kp.left_wrist=gxy("left_wrist",kp.left_wrist)
            kp.right_wrist=gxy("right_wrist",kp.right_wrist)
            kp.left_hip=gxy("left_hip",kp.left_hip)
            kp.right_hip=gxy("right_hip",kp.right_hip)
            kp.left_knee=gxy("left_knee",kp.left_knee)
            kp.right_knee=gxy("right_knee",kp.right_knee)
            kp.left_ankle=gxy("left_ankle",kp.left_ankle)
            kp.right_ankle=gxy("right_ankle",kp.right_ankle)
            nose=g("nose")
            if nose: kp.head=nose
            kp.shoulder_center=((kp.left_shoulder[0]+kp.right_shoulder[0])/2.,
                                  (kp.left_shoulder[1]+kp.right_shoulder[1])/2.)
            kp.hip_center=((kp.left_hip[0]+kp.right_hip[0])/2.,
                            (kp.left_hip[1]+kp.right_hip[1])/2.)
            kp.neck=((kp.shoulder_center[0]+kp.head[0])/2.,
                      (kp.shoulder_center[1]+kp.head[1])/2.)
            for side in ("left","right"):
                ank=getattr(kp,f"{side}_ankle")
                object.__setattr__(kp,f"{side}_foot",(ank[0]+w*.04,ank[1]+h*.03))
        return kp

    def _cwidths(self,frame,bbox):
        bx,by,bw,bh=bbox; H,W=frame.shape[:2]
        bx2=min(bx+bw,W); by2=min(by+bh,H); bx=max(0,bx); by=max(0,by)
        if bx2-bx<5 or by2-by<5: return None
        crop=frame[by:by2,bx:bx2]
        _,mask=cv2.threshold(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ws=np.array([np.sum(mask[r]>0) for r in range(mask.shape[0])],dtype=float)
        return smooth_arr(ws,w=max(3,bh//20)) if len(ws)>5 else None

    def _bwidths(self,cw,bw,bh):
        dsh=bw*.29; dh=bw*.17
        if cw is None or len(cw)<10: return dsh,dh
        n=len(cw); u=cw[int(n*.15):int(n*.40)]; l=cw[int(n*.48):int(n*.68)]
        sh=float(np.max(u))/2. if len(u) else dsh
        hh=float(np.max(l))/2. if len(l) else dh
        return float(np.clip(sh,bw*.18,bw*.42)),float(np.clip(hh,bw*.10,bw*.32))


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN JOINT SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class JointKalman:
    def __init__(self,pn=1.5,on=8.0):
        self.x=None; self.v=0.; self.P=np.array([[100.,0.],[0.,100.]])
        self.Q=np.diag([pn,pn*2]); self.R=on
        self.F=np.array([[1.,1.],[0.,1.]]); self.H=np.array([[1.,0.]])
    def update(self,z):
        if self.x is None: self.x=z; return z
        st=self.F@np.array([self.x,self.v]); Pp=self.F@self.P@self.F.T+self.Q
        y=z-(self.H@st)[0]; S=(self.H@Pp@self.H.T)[0,0]+self.R
        K=Pp@self.H.T/S; st=st+(K*y).flatten()
        self.P=(np.eye(2)-np.outer(K.flatten(),self.H))@Pp
        self.x,self.v=float(st[0]),float(st[1]); return self.x

class PoseKalmanSmoother:
    def __init__(self): self._kx={}; self._ky={}
    def smooth(self,kp):
        out=PoseKeypoints()
        for nm in JOINT_NAMES:
            raw=getattr(kp,nm)
            if nm not in self._kx: self._kx[nm]=JointKalman(); self._ky[nm]=JointKalman()
            object.__setattr__(out,nm,(self._kx[nm].update(raw[0]),self._ky[nm].update(raw[1])))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  SKELETON RENDERER
# ══════════════════════════════════════════════════════════════════════════════

_W=(240,240,240); _L=(255,200,0); _R=(0,140,255); _S=(180,240,180)
BONE_DEFS=[
    ("head","neck",_W,_W,4),("neck","shoulder_center",_W,_S,4),
    ("shoulder_center","hip_center",_S,_S,5),
    ("left_shoulder","left_elbow",_L,_L,5),("left_elbow","left_wrist",_L,_L,4),
    ("right_shoulder","right_elbow",_R,_R,5),("right_elbow","right_wrist",_R,_R,4),
    ("left_shoulder","right_shoulder",_L,_R,4),("left_hip","right_hip",_L,_R,5),
    ("left_hip","left_knee",_L,_L,7),("left_knee","left_ankle",_L,_L,6),("left_ankle","left_foot",_L,_L,4),
    ("right_hip","right_knee",_R,_R,7),("right_knee","right_ankle",_R,_R,6),("right_ankle","right_foot",_R,_R,4),
]

def draw_gradient_bone(img,p1,p2,c1,c2,th,rt=0.):
    s=max(8,int(dist2d(p1,p2)/4))
    for i in range(s):
        t=i/max(s-1,1); t2=(i+1)/max(s-1,1)
        col=lerp_color(lerp_color(c1,c2,t),(0,0,220),rt*.6)
        cv2.line(img,(int(p1[0]+t*(p2[0]-p1[0])),int(p1[1]+t*(p2[1]-p1[1]))),
                 (int(p1[0]+t2*(p2[0]-p1[0])),int(p1[1]+t2*(p2[1]-p1[1]))),col,th,cv2.LINE_AA)

def draw_glow_joint(img,pt,r,col,ga=0.45):
    px,py=int(pt[0]),int(pt[1])
    for rr in range(r+6,r,-2):
        ov=img.copy(); cv2.circle(ov,(px,py),rr,col,-1,cv2.LINE_AA)
        cv2.addWeighted(ov,ga*(1-(rr-r)/6.),img,1-ga*(1-(rr-r)/6.),0,img)
    cv2.circle(img,(px,py),r,(255,255,255),-1,cv2.LINE_AA)
    cv2.circle(img,(px,py),max(1,r-2),col,-1,cv2.LINE_AA)

def render_skeleton(frame,kp,risk_tint=0.):
    kpd={n:getattr(kp,n) for n in JOINT_NAMES}
    for a,b,c1,c2,th in BONE_DEFS:
        if a in kpd and b in kpd: draw_gradient_bone(frame,kpd[a],kpd[b],c1,c2,th,risk_tint)
    sz={"head":4,"neck":3,"left_shoulder":4,"right_shoulder":4,"left_elbow":3,"right_elbow":3,
        "left_wrist":3,"right_wrist":3,"left_hip":5,"right_hip":5,"left_knee":6,"right_knee":6,
        "left_ankle":5,"right_ankle":5,"left_foot":3,"right_foot":3}
    for nm,r in sz.items():
        if nm in kpd:
            col=lerp_color(_L if "left" in nm else _R if "right" in nm else _W,(0,0,220),risk_tint*.5)
            draw_glow_joint(frame,kpd[nm],r,col)

def draw_risk_gauge(frame,cx,cy,radius,s,label="RISK"):
    cv2.ellipse(frame,(cx,cy),(radius,radius),225,0,270,(30,30,30),6,cv2.LINE_AA)
    sw=int(270*clamp01(s/100.))
    if sw>0: cv2.ellipse(frame,(cx,cy),(radius,radius),225,0,sw,risk_color(s),6,cv2.LINE_AA)
    col=risk_color(s)
    cv2.putText(frame,f"{s:.0f}",(cx-18,cy+7),cv2.FONT_HERSHEY_SIMPLEX,.75,col,2,cv2.LINE_AA)
    cv2.putText(frame,label,(cx-20,cy+radius-2),cv2.FONT_HERSHEY_SIMPLEX,.38,(180,180,180),1,cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class SportsAnalyzer:
    PIX_TO_M=None

    def __init__(self,video_path,output_video_path="output_annotated.mp4",
                 player_id=1,fps_override=None,pick=False,yolo_size="m"):
        self.video_path=video_path; self.output_video_path=output_video_path
        self.player_id=player_id; self.fps_override=fps_override
        self.pose_est=HybridPoseEstimator(); self.smoother=PoseKalmanSmoother()
        self.pose_frames:List[PoseFrame]=[]; self.frame_metrics:List[FrameMetrics]=[]
        self.summary=PlayerSummary(player_id=player_id)
        self._spd_win=deque(maxlen=30); self._risk_win=deque(maxlen=15)
        self._trail=deque(maxlen=60); self._speed_history=deque(maxlen=90)
        self._accel_burst=0; self._fps_cache=30.
        _get_detection_layer(yolo_size)
        if pick:
            print("[INFO] Interactive selection …"); primary=pick_player_interactive(video_path)
        else:
            print("[INFO] Pre-scan …"); primary=select_primary_player(video_path)
        if primary is None: raise RuntimeError("No player candidates found.")
        self.lock=TargetLock(primary["seed_bbox"],primary["hist"],primary["seed_frame"])

    def process_video(self):
        cap=cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): raise FileNotFoundError(self.video_path)
        fps=self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.
        self._fps_cache=fps
        W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out=cv2.VideoWriter(self.output_video_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
        print(f"[INFO] {total} frames @ {fps:.1f} fps  ({W}×{H})")
        idx=0
        while True:
            ret,frame=cap.read()
            if not ret: break
            ts=idx/fps; bbox=self.lock.update(frame)
            visible=False
            if bbox and bbox[2]>20 and bbox[3]>40:
                visible=True; target=next((t for t in self.lock.bt.active_tracks if t.id==self.lock._target_id),None)
                yolo_kp=getattr(target,'_yolo_kp',None) if target else None; spd=self.frame_metrics[-1].speed if self.frame_metrics else 0.
                raw_kp=self.pose_est.estimate(frame,bbox,ts,spd,yolo_kp=yolo_kp); kp=self.smoother.smooth(raw_kp); pf=PoseFrame(idx,ts,bbox,kp); self.pose_frames.append(pf)
                if self.PIX_TO_M is None: self._calibrate(kp)
                fm=self._metrics(pf,idx,ts,fps); self.frame_metrics.append(fm); self._trail.append((int(kp.hip_center[0]),int(kp.hip_center[1]))); self._speed_history.append(fm.speed)
                if abs(fm.acceleration)>4.0: self._accel_burst=8
                elif self._accel_burst>0: self._accel_burst-=1
                frame=self._draw_trail(frame); frame=self._annotate(frame,pf,fm,W,H); frame=self._draw_player_aura(frame,kp,fm)
            frame=self._hud(frame,idx,ts,total,visible); out.write(frame); idx+=1
        cap.release(); out.release(); tr=len(self.pose_frames); print(f"[INFO] Tracked {tr}/{idx} frames ({100*tr/max(idx,1):.1f}%)")
        self._post_gait(fps); self._build_summary(); print("[INFO] Done."); return self.summary

    def _calibrate(self,kp):
        leg=(dist2d(kp.left_hip,kp.left_ankle)+dist2d(kp.right_hip,kp.right_ankle))/2; self.PIX_TO_M=0.9/leg if leg>10 else 0.002

    def _metrics(self,pf,idx,ts,fps)->FrameMetrics:
        fm=FrameMetrics(frame_idx=idx,timestamp=ts); kp=pf.kp; sc=self.PIX_TO_M or 0.002
        fm.left_knee_angle=angle_3pts(kp.left_hip,kp.left_knee,kp.left_ankle); fm.right_knee_angle=angle_3pts(kp.right_hip,kp.right_knee,kp.right_ankle)
        fm.left_hip_angle=angle_3pts(kp.left_shoulder,kp.left_hip,kp.left_knee); fm.right_hip_angle=angle_3pts(kp.right_shoulder,kp.right_hip,kp.right_knee)
        dx=kp.shoulder_center[0]-kp.hip_center[0]; dy=kp.shoulder_center[1]-kp.hip_center[1]; fm.trunk_lean=math.degrees(math.atan2(abs(dx),abs(dy)+1e-9))
        hw=dist2d(kp.left_hip,kp.right_hip)+1e-6; fm.l_valgus=abs(kp.left_knee[0]-kp.left_hip[0])/hw; fm.r_valgus=abs(kp.right_knee[0]-kp.right_hip[0])/hw
        if len(self.pose_frames)>=2:
            prev=self.pose_frames[-2]; dt=ts-prev.timestamp+1e-9; dp=dist2d(kp.hip_center,prev.kp.hip_center)*sc; raw=dp/dt
            self._spd_win.append(raw); fm.speed=float(np.mean(self._spd_win)); fm.body_center_disp=dp
            if len(self.pose_frames)>=3:
                p2=self.pose_frames[-3]; dp2=dist2d(prev.kp.hip_center,p2.kp.hip_center)*sc
                dt2=prev.timestamp-p2.timestamp+1e-9; fm.acceleration=(raw-dp2/dt2)/dt
        if len(self.pose_frames)>=5:
            pos=[p.kp.hip_center for p in list(self.pose_frames)[-5:]]; vecs=[(pos[i+1][0]-pos[i][0],pos[i+1][1]-pos[i][1]) for i in range(4)]
            for i in range(len(vecs)-1):
                v1,v2=np.array(vecs[i]),np.array(vecs[i+1]); n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
                if n1>2 and n2>2 and math.acos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))>math.radians(28): fm.direction_change=True
        fm.energy_expenditure=max(1.5,3.5+fm.speed*2.3)*75; ks=sum((155-a)/155 for a in [fm.left_knee_angle,fm.right_knee_angle] if a<155); fm.joint_stress=min(1.,ks/2)
        if len(self._spd_win)>=10:
            s=list(self._spd_win); fm.fatigue_index=max(0.,min(1.,(np.mean(s[:5])-np.mean(s[-5:]))/(np.mean(s[:5])+1e-6)))
        pv=clamp01(((fm.l_valgus+fm.r_valgus)/2-.02)/.08); pk=clamp01(abs(fm.left_knee_angle-fm.right_knee_angle)/30.); pa=clamp01(abs(fm.acceleration)/12); pt=clamp01(fm.trunk_lean/30)
        ls = clamp01(fm.trunk_lean / 25.) * (max(0, fm.speed - 1.0) / 5.0); asym = clamp01(abs(fm.left_knee_angle-fm.right_knee_angle)/40.)
        fm.joint_stress = clamp01(fm.joint_stress * 0.5 + ls * 0.3 + asym * 0.2); rr=.30*pv+.25*fm.joint_stress+.20*pk+.15*pa+.10*pt
        self._risk_win.append(rr); fm.risk_score=float(np.mean(self._risk_win))*100.; fm.fall_risk=0.; fm.injury_risk=rr; return fm


    def _post_gait(self,fps):
        if len(self.pose_frames)<15: return
        sc=self.PIX_TO_M or 0.002; la=smooth_arr([p.kp.left_ankle[1] for p in self.pose_frames]); ra=smooth_arr([p.kp.right_ankle[1] for p in self.pose_frames]); md=max(5,int(fps*.18))
        if HAS_SCIPY: lp,_=find_peaks(la,distance=md,prominence=2); rp,_=find_peaks(ra,distance=md,prominence=2)
        else:
            def pk(arr,d):
                pks=[]
                for i in range(1,len(arr)-1):
                    if arr[i]>arr[i-1] and arr[i]>arr[i+1]:
                        if not pks or i-pks[-1]>=d: pks.append(i)
                return np.array(pks)
            lp=pk(la,md); rp=pk(ra,md)
        pos=[p.kp.hip_center for p in self.pose_frames]; strl,stt,flt=[],[],[]
        for peaks in [lp,rp]:
            for i in range(1,len(peaks)):
                i0,i1=peaks[i-1],peaks[i]
                if i1>=len(pos): continue
                sl=dist2d(pos[i0],pos[i1])*sc
                if .15<sl<3.5: strl.append(sl)
                st=(i1-i0)/fps
                if .08<st<2.0: stt.append(st); flt.append(max(0.,st*.35))
        n=min(len(lp),len(rp))-1
        if n>0:
            li=[(lp[i+1]-lp[i])/fps for i in range(n)]; ri=[(rp[i+1]-rp[i])/fps for i in range(n)]; m=min(len(li),len(ri))
            sym=float(np.mean([1-abs(l-r)/(l+r+1e-9) for l,r in zip(li[:m],ri[:m])]))*100
        else: sym=94.
        sv=float(np.std(strl)/(np.mean(strl)+1e-9)*100) if len(strl)>2 else 3.5; asl=float(np.mean(strl)) if strl else 1.35; ast=float(np.mean(stt)) if stt else .38; aft=float(np.mean(flt)) if flt else .13; acad=60./ast if ast>0 else 158.
        for fm in self.frame_metrics: fm.stride_length=asl; fm.step_time=ast; fm.flight_time=aft; fm.cadence=acad; fm.gait_symmetry=sym; fm.stride_variability=sv
        hip_x = [p.kp.hip_center[0] for p in self.pose_frames]; lat_bal = clamp01(np.std(hip_x) / (max(1, np.mean([p.bbox[2] for p in self.pose_frames]))*0.1) if hip_x else 0.)
        for fm in self.frame_metrics:
            sr=max(0.,(100-fm.gait_symmetry)/100); vr=min(1.,fm.stride_variability/25); lr=min(1.,fm.trunk_lean/40); fm.fall_risk = clamp01(sr*.3 + vr*.2 + lr*.2 + lat_bal*.3); ar=min(1.,abs(fm.acceleration)/12); fm.injury_risk=fm.joint_stress*.5+ar*.3+fm.fatigue_index*.2

    def _draw_trail(self,frame):
        pts=list(self._trail)
        if len(pts)<2: return frame
        ov=frame.copy()
        for i in range(1,len(pts)): t=i/len(pts); col=lerp_color((40,40,120),(0,220,255),t); cv2.line(ov,pts[i-1],pts[i],col,max(1,int(t*4)),cv2.LINE_AA)
        cv2.circle(ov,pts[0],3,(80,80,180),-1,cv2.LINE_AA); frame[:]=cv2.addWeighted(ov,.55,frame,.45,0); return frame

    def _annotate(self,frame,pf,fm,W,H):
        kp=pf.kp; rt=clamp01(fm.risk_score/100.); render_skeleton(frame,kp,risk_tint=rt); hx,hy=int(kp.hip_center[0]),int(kp.hip_center[1]); head_y=int(kp.head[1])-28; badge=f"  #{self.player_id}  "; (tw,th),_=cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,.62,2); bx0=hx-tw//2-4
        cv2.rectangle(frame,(bx0,head_y-20),(bx0+tw+8,head_y+4),(255,220,0),-1); cv2.rectangle(frame,(bx0,head_y-20),(bx0+tw+8,head_y+4),(0,0,0),1); cv2.putText(frame,badge,(bx0+4,head_y),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,0,0),2,cv2.LINE_AA)
        if fm.speed>.3 and len(self.pose_frames)>=2:
            prev=self.pose_frames[-2]; dx=kp.hip_center[0]-prev.kp.hip_center[0]; dy=kp.hip_center[1]-prev.kp.hip_center[1]; mag=math.hypot(dx,dy)+1e-6; ar=int(min(fm.speed*14,90)); ex,ey=int(hx+dx/mag*ar),int(hy+dy/mag*ar); sc2=risk_color(fm.speed/10.*100); cv2.arrowedLine(frame,(hx,hy),(ex,ey),sc2,4,cv2.LINE_AA,tipLength=.32); cv2.arrowedLine(frame,(hx,hy),(ex,ey),(255,255,255),1,cv2.LINE_AA,tipLength=.32); spdt=f"{fm.speed:.1f} m/s"; (stw,sth),_=cv2.getTextSize(spdt,cv2.FONT_HERSHEY_SIMPLEX,.52,2); cv2.rectangle(frame,(ex+4,ey-sth-4),(ex+stw+8,ey+4),(0,0,0),-1); cv2.putText(frame,spdt,(ex+6,ey),cv2.FONT_HERSHEY_SIMPLEX,.52,sc2,2,cv2.LINE_AA)
        for kpt,ang in [(kp.left_knee,fm.left_knee_angle),(kp.right_knee,fm.right_knee_angle)]:
            kx,ky=int(kpt[0]),int(kpt[1]); ac=(0,220,0) if ang>145 else (0,140,255) if ang>120 else (0,0,220)
            if ang<130:
                pulse=int(8+3*math.sin(len(self.pose_frames)*.4)); ov=frame.copy(); cv2.circle(ov,(kx,ky),pulse,(0,0,255),2,cv2.LINE_AA); frame[:]=cv2.addWeighted(ov,.6,frame,.4,0)
            cv2.putText(frame,f"{ang:.0f}°",(kx+8,ky-5),cv2.FONT_HERSHEY_SIMPLEX,.44,ac,1,cv2.LINE_AA)
        if fm.direction_change:
            ov=frame.copy(); cv2.rectangle(ov,(0,0),(W,H),(0,0,180),6); frame[:]=cv2.addWeighted(ov,.55,frame,.45,0); lbl="DIRECTION CHANGE"; (lw,_),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,1.1,2); cv2.rectangle(frame,(W//2-lw//2-10,H-52),(W//2+lw//2+10,H-16),(0,0,0),-1); cv2.putText(frame,lbl,(W//2-lw//2,H-24),cv2.FONT_HERSHEY_DUPLEX,1.1,(0,100,255),2,cv2.LINE_AA)
        draw_risk_gauge(frame,W-70,H-80,44,fm.risk_score); return frame

    def _draw_player_aura(self,frame,kp,fm):
        if fm.speed<.5: return frame
        hx,hy=int(kp.hip_center[0]),int(kp.hip_center[1]); bx,by,bw,bh=self.pose_frames[-1].bbox; rx=max(12,bw//2+6); ry=max(20,bh//2+10); col=lerp_color((0,180,60),(0,60,255),clamp01(fm.speed/8.))
        if self._accel_burst>0:
            br=int(rx*1.6+self._accel_burst*3); ba=self._accel_burst/8.*.4; ov=frame.copy(); cv2.ellipse(ov,(hx,hy),(br,int(br*1.4)),0,0,360,(0,200,255),3,cv2.LINE_AA); frame[:]=cv2.addWeighted(ov,ba,frame,1-ba,0)
        for exp,a in [(14,.12),(6,.20)]:
            ov=frame.copy(); cv2.ellipse(ov,(hx,hy),(rx+exp,ry+exp),0,0,360,col,-1,cv2.LINE_AA); frame[:]=cv2.addWeighted(ov,a,frame,1-a,0)
        return frame

    def _draw_stat_bar(self,frame,x,y,w,h,val,mx,col,lbl,fmt):
        filled=int(w*clamp01(val/max(mx,1e-6))); cv2.rectangle(frame,(x,y),(x+w,y+h),(35,35,35),-1)
        for px in range(filled): t=px/max(w-1,1); cv2.line(frame,(x+px,y+1),(x+px,y+h-1),lerp_color(col,lerp_color(col,(255,255,255),.4),t),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(60,60,60),1); cv2.putText(frame,lbl,(x-2,y+h-2),cv2.FONT_HERSHEY_SIMPLEX,.34,(160,160,160),1,cv2.LINE_AA)
        vt=fmt.format(val); (vw,_),_=cv2.getTextSize(vt,cv2.FONT_HERSHEY_SIMPLEX,.38,1); cv2.putText(frame,vt,(x+w+4,y+h-2),cv2.FONT_HERSHEY_SIMPLEX,.38,col,1,cv2.LINE_AA)

    def _draw_sparkline(self,frame,x,y,w,h,vals,col=(0,255,200)):
        v=list(vals)
        if len(v)<2: return
        cv2.rectangle(frame,(x,y),(x+w,y+h),(20,20,20),-1); cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,50),1); mx=max(max(v),.1); pts=[(x+int(i/(len(v)-1)*w),y+h-int(clamp01(vi/mx)*(h-2))-1) for i,vi in enumerate(v)]
        for i in range(1,len(pts)): cv2.line(frame,pts[i-1],pts[i],lerp_color((0,120,100),col,i/len(pts)),2,cv2.LINE_AA)
        cv2.circle(frame,pts[-1],3,(255,255,255),-1,cv2.LINE_AA); cv2.putText(frame,f"{v[-1]:.1f}",(x+w+3,y+h),cv2.FONT_HERSHEY_SIMPLEX,.34,col,1,cv2.LINE_AA)

    def _hud(self,frame,idx,ts,total,visible=True):
        H,W=frame.shape[:2]; fps=self._fps_cache; PW=230; ov=frame.copy(); cv2.rectangle(ov,(0,0),(PW,H),(8,8,12),-1); frame[:]=cv2.addWeighted(ov,.72,frame,.28,0); cv2.line(frame,(PW-1,0),(PW-1,H),(255,215,0),2); cv2.rectangle(frame,(0,0),(PW,42),(20,20,28),-1); cv2.line(frame,(0,42),(PW,42),(255,215,0),1); cv2.rectangle(frame,(6,6),(34,36),(255,255,255),-1); cv2.rectangle(frame,(20,6),(34,36),(0,0,0),-1); cv2.putText(frame,"JUV",(7,30),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1); cv2.putText(frame,"ANALYTICS",(38,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,215,0),1,cv2.LINE_AA); dm=_detection_layer.mode.upper() if _detection_layer else "BLOB"; cv2.putText(frame,f"SPORTS SCIENCE  [{dm}]",(38,34),cv2.FONT_HERSHEY_SIMPLEX,.28,(120,120,120),1,cv2.LINE_AA); ts_=self.lock.state; dc={"searching":(0,200,255),"tracking":(0,220,0),"lost":(0,80,255)}.get(ts_,(150,150,150)); st={"searching":"ACQUIRING","tracking":"LIVE","lost":"SEARCHING"}.get(ts_,""); cv2.circle(frame,(10,54),5,dc,-1,cv2.LINE_AA); cv2.putText(frame,st,(20,59),cv2.FONT_HERSHEY_SIMPLEX,.38,dc,1,cv2.LINE_AA); tc=f"{int(ts//60):02d}:{ts%60:05.2f}"; (tcw,_),_=cv2.getTextSize(tc,cv2.FONT_HERSHEY_SIMPLEX,.38,1); cv2.putText(frame,tc,(PW-tcw-6,59),cv2.FONT_HERSHEY_SIMPLEX,.38,(160,160,160),1,cv2.LINE_AA); cv2.rectangle(frame,(0,66),(PW,92),(18,18,26),-1); cv2.putText(frame,f"PLAYER  #{self.player_id}",(8,84),cv2.FONT_HERSHEY_DUPLEX,.52,(255,215,0),1,cv2.LINE_AA)
        if self.frame_metrics:
            fm=self.frame_metrics[-1]; rc=risk_color(fm.risk_score); y0=100; cv2.putText(frame,"VELOCITY",(8,y0),cv2.FONT_HERSHEY_SIMPLEX,.33,(120,120,120),1,cv2.LINE_AA); cv2.line(frame,(0,y0+3),(PW,y0+3),(30,30,40),1); spdt=f"{fm.speed:.1f}"; cv2.putText(frame,spdt,(8,y0+38),cv2.FONT_HERSHEY_DUPLEX,1.4,(0,255,200),2,cv2.LINE_AA); cv2.putText(frame,"m/s",(8+int(len(spdt)*22),y0+38),cv2.FONT_HERSHEY_SIMPLEX,.42,(80,200,160),1,cv2.LINE_AA); self._draw_sparkline(frame,8,y0+44,PW-20,28,self._speed_history); cv2.putText(frame,"3s SPEED HISTORY",(8,y0+84),cv2.FONT_HERSHEY_SIMPLEX,.28,(80,80,80),1,cv2.LINE_AA); y1=y0+96; BW=80; BH=7; BX=68
            for i,(lbl,val,mx,col,fmt) in enumerate([("CADENCE",fm.cadence,220.,(0,200,255),"{:.0f} spm"),("STRIDE",fm.stride_length,2.5,(0,255,180),"{:.2f} m"),("ENERGY",fm.energy_expenditure,900.,(0,180,255),"{:.0f} kc"),("GAIT SYM",fm.gait_symmetry,100.,(0,255,150),"{:.0f}%"),("TRUNK",fm.trunk_lean,30.,(0,200,220),"{:.1f}°")]): sy=y1+i*22; cv2.putText(frame,lbl,(6,sy+BH),cv2.FONT_HERSHEY_SIMPLEX,.30,(110,110,110),1,cv2.LINE_AA); self._draw_stat_bar(frame,BX,sy,BW,BH,val,mx,col,"",fmt)
            y2=y1+5*22+8; cv2.putText(frame,"JOINT ANGLES",(8,y2),cv2.FONT_HERSHEY_SIMPLEX,.33,(120,120,120),1,cv2.LINE_AA); cv2.line(frame,(0,y2+3),(PW,y2+3),(30,30,40),1)
            for ki,(s,ang) in enumerate([("L KNEE",fm.left_knee_angle),("R KNEE",fm.right_knee_angle)]): sy=y2+14+ki*20; ac=(0,220,0) if ang>145 else (0,140,255) if ang>120 else (0,0,220); cv2.putText(frame,s,(6,sy+10),cv2.FONT_HERSHEY_SIMPLEX,.32,(110,110,110),1,cv2.LINE_AA); self._draw_stat_bar(frame,BX,sy,BW,BH,ang,180.,ac,"","{:.0f}°")
            y3=y2+14+2*20+6; cv2.putText(frame,"VALGUS",(8,y3),cv2.FONT_HERSHEY_SIMPLEX,.33,(120,120,120),1,cv2.LINE_AA); cv2.line(frame,(0,y3+3),(PW,y3+3),(30,30,40),1)
            for ki,(s,val) in enumerate([("L",fm.l_valgus),("R",fm.r_valgus)]): sy=y3+14+ki*20; vc=(0,220,0) if val<.1 else (0,140,255) if val<.2 else (0,0,220); cv2.putText(frame,s,(6,sy+10),cv2.FONT_HERSHEY_SIMPLEX,.32,(110,110,110),1,cv2.LINE_AA); self._draw_stat_bar(frame,BX,sy,BW,BH,val,.4,vc,"","{:.2f}")
            y4=y3+14+2*20+8; cv2.putText(frame,"INJURY RISK",(8,y4),cv2.FONT_HERSHEY_SIMPLEX,.33,(120,120,120),1,cv2.LINE_AA); cv2.line(frame,(0,y4+3),(PW,y4+3),(30,30,40),1); rt=f"{fm.risk_score:.0f}"; cv2.putText(frame,rt,(8,y4+38),cv2.FONT_HERSHEY_DUPLEX,1.4,rc,2,cv2.LINE_AA); cv2.putText(frame,"/ 100",(8+int(len(rt)*22),y4+38),cv2.FONT_HERSHEY_SIMPLEX,.4,(80,80,80),1,cv2.LINE_AA); rl=self._risk_label(fm.injury_risk); lc=(0,200,0) if rl=="Low" else (0,140,255) if rl=="Moderate" else (0,0,220); (rlw,_),_=cv2.getTextSize(rl,cv2.FONT_HERSHEY_SIMPLEX,.42,1); rx0=8; ry0=y4+44; cv2.rectangle(frame,(rx0,ry0),(rx0+rlw+10,ry0+18),lc,-1); cv2.putText(frame,rl,(rx0+5,ry0+13),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA)
            for ki,(lbl,val,mx,col) in enumerate([("FALL",fm.fall_risk,1.,(0,200,255)),("JOINT",fm.joint_stress,1.,(0,140,255)),("FATIGUE",fm.fatigue_index,1.,(0,100,220))]): sy=y4+68+ki*20; cv2.putText(frame,lbl,(6,sy+8),cv2.FONT_HERSHEY_SIMPLEX,.28,(100,100,100),1,cv2.LINE_AA); self._draw_stat_bar(frame,BX,sy,BW,6,val,mx,risk_color(val*100),"","{:.2f}")
        prog=clamp01(ts/max(total/fps,1e-6)); pby=H-18; cv2.rectangle(frame,(0,pby),(PW,H),(15,15,20),-1)
        if prog>0: cv2.rectangle(frame,(0,pby+2),(int(PW*prog),H-2),(255,215,0),-1); cv2.putText(frame,f"{prog*100:.0f}%",(PW//2-10,H-4),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,0,0) if prog>.3 else (120,120,120),1,cv2.LINE_AA)
        if self.frame_metrics:
            rc2=risk_color(self.frame_metrics[-1].risk_score); bw=int((W-PW)*clamp01(self.frame_metrics[-1].risk_score/100.)); cv2.rectangle(frame,(PW,H-8),(W,H),(20,20,20),-1)
            if bw>0: cv2.rectangle(frame,(PW,H-8),(PW+bw,H),rc2,-1)
        if not visible and ts_=="lost":
            bov=frame.copy(); cv2.rectangle(bov,(PW,H//2-35),(W,H//2+35),(0,0,0),-1); frame[:]=cv2.addWeighted(bov,.70,frame,.30,0); cv2.putText(frame,f"PLAYER #{self.player_id} — OUT OF FRAME",(PW+20,H//2+8),cv2.FONT_HERSHEY_DUPLEX,.75,(0,140,255),2,cv2.LINE_AA)
        return frame

    def _build_summary(self):
        if not self.frame_metrics: return
        fms=self.frame_metrics; s=self.summary; sc=self.PIX_TO_M or 0.002; s.total_frames=len(fms); s.duration_seconds=fms[-1].timestamp; spds=np.array([f.speed for f in fms]); s.avg_speed=float(np.mean(spds)); s.max_speed=float(np.max(spds))
        def anz(a): v=[getattr(f,a) for f in fms if getattr(f,a)>0]; return float(np.mean(v)) if v else 0.
        s.avg_stride_length=anz("stride_length"); s.avg_step_time=anz("step_time"); s.avg_cadence=anz("cadence"); s.avg_flight_time=anz("flight_time"); s.estimated_energy_kcal_hr=float(np.mean([f.energy_expenditure for f in fms])); s.gait_symmetry_pct=float(np.mean([f.gait_symmetry for f in fms])); s.stride_variability_pct=float(np.mean([f.stride_variability for f in fms]))
        dc=sum(1 for f in fms if f.direction_change); s.direction_change_freq=dc/max(s.duration_seconds/60,1e-6); s.peak_risk_score=float(np.max([f.risk_score for f in fms]))
        if len(self.pose_frames)>=2: s.total_distance_m=sum(dist2d(self.pose_frames[i].kp.hip_center,self.pose_frames[i-1].kp.hip_center)*sc for i in range(1,len(self.pose_frames)))
        def rl(a): return self._risk_label(float(np.mean([getattr(f,a) for f in fms]))); s.fall_risk_label=rl("fall_risk"); s.injury_risk_label=rl("injury_risk"); s.body_stress_label=rl("joint_stress"); s.fatigue_label=rl("fatigue_index"); ai=float(np.mean([f.injury_risk for f in fms])); s.injury_risk_detail="high knee load" if ai>.5 else "moderate joint stress" if ai>.3 else "within normal range"

    def get_report_string(self) -> str:
        s = self.summary
        dm = _detection_layer.mode.upper() if _detection_layer else "BLOB"
        
        lines = []
        width = 70
        
        # Header
        lines.append("=" * width)
        lines.append(f"JUVENTUS ANALYTICS v4 — Player #{s.player_id} [{dm}]".center(width))
        lines.append("=" * width)
        
        # Session Overview
        lines.append("")
        lines.append("SESSION OVERVIEW")
        lines.append("-" * width)
        lines.append(f"  Duration        : {s.duration_seconds:>6.1f} s")
        lines.append(f"  Total Frames    : {s.total_frames:>6}")
        lines.append(f"  Total Distance  : {s.total_distance_m:>6.1f} m")
        
        # Player Metrics
        lines.append("")
        lines.append("PLAYER METRICS")
        lines.append("-" * width)
        lines.append(f"  Avg Speed       : {s.avg_speed:>6.2f} m/s")
        lines.append(f"  Max Speed       : {s.max_speed:>6.2f} m/s")
        lines.append(f"  Avg Stride      : {s.avg_stride_length:>6.2f} m")
        lines.append(f"  Avg Cadence     : {s.avg_cadence:>6.0f} spm")
        lines.append(f"  Avg Step Time   : {s.avg_step_time:>6.2f} s")
        lines.append(f"  Changes/Min     : {s.direction_change_freq:>6.1f}")
        lines.append(f"  Energy Burn     : {s.estimated_energy_kcal_hr:>6.0f} kcal/hr")
        
        # Risk Indicators
        lines.append("")
        lines.append("RISK INDICATORS")
        lines.append("-" * width)
        lines.append(f"  Peak Risk Score : {s.peak_risk_score:>6.0f} / 100")
        lines.append(f"  Gait Symmetry   : {s.gait_symmetry_pct:>6.1f} %")
        lines.append(f"  Fall Risk       : {s.fall_risk_label:<10}")
        lines.append(f"  Injury Risk     : {s.injury_risk_label:<10}")
        lines.append(f"  Body Stress     : {s.body_stress_label:<10}")
        lines.append(f"  Fatigue Level   : {s.fatigue_label:<10}")
        lines.append(f"  Risk Detail     : {s.injury_risk_detail:<20}")
        
        lines.append("")
        lines.append("=" * width)
        
        return "\n".join(lines)

    @staticmethod
    def _risk_label(v): 
        return "Low" if v < .25 else "Moderate" if v < .55 else "High"

    def export_json(self, path):
        with open(path, "w") as f:
            json.dump({
                "player_summary": asdict(self.summary),
                "frame_metrics": [asdict(m) for m in self.frame_metrics]
            }, f, indent=2)
        print(f"[EXPORT] JSON → {path}")

    def export_csv(self, path):
        pd.DataFrame([asdict(m) for m in self.frame_metrics]).to_csv(path, index=False)
        print(f"[EXPORT] CSV  → {path}")

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(m) for m in self.frame_metrics])

    def print_report(self):
        # Use the centralized report string for consistent formatting
        print(self.get_report_string())
