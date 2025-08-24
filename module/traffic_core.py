# traffic_core.py
import os, json, math
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ---------- Helpers ----------
def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def clip_box(x1, y1, x2, y2, W, H):
    return max(0, int(x1)), max(0, int(y1)), min(W-1, int(x2)), min(H-1, int(y2))

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / max(1e-6, (area_a + area_b - inter))

def variance_of_laplacian(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def post_format_plate(s: str) -> str:
    """
    Chuyển đổi biển số xe theo quy tắc Việt Nam
    - Chuyển O -> 0 (biển số VN không có chữ O)
    - Lưu ý: Chuyển đổi 8->B, 2->Z đã được xử lý trong PlateTextFuser
    """
    s = "".join(ch for ch in s if ch.isalnum()).upper()
    if not s:
        return ""
    
    # 1. Chữ "O" luôn chuyển thành "0" (biển số VN không có chữ O)
    s = s.replace("O", "0")
    
    return s

# ---------- HelmetEMA ----------
class HelmetEMA:
    def __init__(self, alpha=0.7, yes_thr=0.6, no_thr=0.4):
        self.alpha = float(alpha)
        self.yes_thr = float(yes_thr)
        self.no_thr = float(no_thr)
        self.ema = 0.7
        self.count = 0

    def update(self, prob: float, conf: float = 1.0):
        # EMA có trọng số theo độ tự tin: w = (1 - alpha) * conf
        # prob kỳ vọng trong [0,1]
        try:
            p = float(prob)
        except Exception:
            p = 0.0
        p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
        try:
            c = float(conf)
        except Exception:
            c = 1.0
        c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)

        w = (1.0 - self.alpha) * c
        self.ema = (1.0 - w) * self.ema + w * p
        self.count += 1
        return self.ema

    @property
    def state(self) -> str:
        # Quy ước mới: > yes_thr => helmet; ngược lại => no_helmet
        return "helmet" if self.ema > self.yes_thr else "no_helmet"

# ---------- PlateTextFuser ----------
class PlateTextFuser:
    def __init__(self, max_slots=12, decay=0.9, entropy_lock=0.6, min_repeats=3):
        self.max_slots = int(max_slots)
        self.decay = float(decay)
        self.entropy_lock = float(entropy_lock)
        self.min_repeats = int(min_repeats)
        self.alphabet = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.hist = [defaultdict(float) for _ in range(self.max_slots)]
        self.len_hist = defaultdict(float)
        self.reads = 0
        self._locked = False
        self._final = ""
        self._final_conf = 0.0
        self._last_candidate = ""
        self._same_in_row = 0

    def _decay_all(self):
        for d in self.hist:
            for k in list(d.keys()):
                d[k] *= self.decay
                if d[k] < 1e-5: del d[k]
        for k in list(self.len_hist.keys()):
            self.len_hist[k] *= self.decay
            if self.len_hist[k] < 1e-5: del self.len_hist[k]

    def update(self, text: str, confs: Optional[List[float]] = None):
        if self._locked or not text:
            return
        s = "".join(ch for ch in text.upper() if ch.isalnum())
        if not s: return
        # Áp dụng logic chuyển đổi ký tự thứ 3: 8 -> B, 2 -> Z
        if len(s) >= 3 and s[2] == "8":
            s = s[:2] + "B" + s[3:]  # 8 -> B
        elif len(s) >= 3 and s[2] == "2":
            s = s[:2] + "Z" + s[3:]  # 2 -> Z
        L = min(len(s), self.max_slots)
        self._decay_all()
        self.len_hist[L] += 1.0
        if not confs or not hasattr(confs, "__len__"):
            confs = [1.0] * L
        for i in range(L):
            ch = s[i]
            if ch not in self.alphabet: continue
            wi = float(confs[i] if i < len(confs) else confs[-1])
            wi = max(0.05, min(1.0, wi))
            self.hist[i][ch] += wi
        self.reads += 1

        cand = self.candidate()
        if cand == self._last_candidate and cand:
            self._same_in_row += 1
        else:
            self._same_in_row = 1
            self._last_candidate = cand
        if self._same_in_row >= self.min_repeats and self.avg_entropy() <= self.entropy_lock:
            self._locked = True
            self._final = cand
            confs = []
            for i,c in enumerate(cand):
                votes = self.hist[i]
                tot = sum(votes.values()) + 1e-6
                confs.append(votes.get(c, 0.0)/tot)
            self._final_conf = float(np.mean(confs)) if confs else 0.0

    def candidate(self) -> str:
        if self._locked: return self._final
        if not self.len_hist: return ""
        L = int(max(self.len_hist.items(), key=lambda kv: kv[1])[0])
        chars = []
        for i in range(L):
            votes = self.hist[i]
            if not votes: break
            ch = max(votes.items(), key=lambda kv: kv[1])[0]
            chars.append(ch)
        return "".join(chars)

    def avg_entropy(self) -> float:
        Hs = []
        for votes in self.hist:
            tot = sum(votes.values())
            if tot <= 0: continue
            probs = np.array([v/tot for v in votes.values()], dtype=float)
            H = -(probs * np.log2(np.clip(probs, 1e-9, 1))).sum()
            Hs.append(H / math.log2(len(self.alphabet)))
        return float(np.mean(Hs)) if Hs else 1.0

    @property
    def locked(self): return self._locked
    @property
    def final_text(self): return self._final
    @property
    def final_conf(self): return self._final_conf

# ---------- SnapshotSelector ----------
class SnapshotSelector:
    def __init__(self, improvement_delta=0.15):
        self.improve_delta = float(improvement_delta)
        self.best_scores = {"motorbike": 0.0, "helmet": 0.0, "lp_img": 0.0}
        self.best_paths = {"motorbike": None, "helmet": None, "lp_img": None}

    def _score(self, frame_bgr, bbox, kind: str, parent_bbox=None):
        H, W = frame_bgr.shape[:2]
        x1,y1,x2,y2 = clip_box(*bbox, W, H)
        if x2 <= x1 or y2 <= y1:
            return 0.0, None
        crop = frame_bgr[y1:y2, x1:x2]
        area_norm = ((x2-x1)*(y2-y1)) / float(W*H + 1e-6)
        blur = variance_of_laplacian(crop)
        blur_n = min(1.0, blur / 3000.0)
        ar = (x2-x1)/max(1, (y2-y1))
        aspect_ok = math.exp(-((ar-4.0)**2)/(2*1.2**2)) if kind == "lp_img" else 1.0
        if parent_bbox is not None:
            px1,py1,px2,py2 = parent_bbox
            pcx, pcy = (px1+px2)/2.0, (py1+py2)/2.0
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            dx = (cx-pcx)/(max(10, px2-px1))
            dy = (cy-pcy)/(max(10, py2-py1))
            center_w = math.exp(-3*(dx*dx + dy*dy))
        else:
            center_w = 1.0
        score = (0.5*blur_n + 0.3*area_norm + 0.2*aspect_ok) * center_w
        return float(score), crop

    def maybe_update_and_save(self, out_dir: str, frame_bgr, bbox, kind: str, parent_bbox=None):
        ensure_dir(out_dir)
        score, crop = self._score(frame_bgr, bbox, kind, parent_bbox)
        if crop is None: return None, 0.0, False
        best = self.best_scores.get(kind, 0.0)
        improved = (score > best * (1.0 + self.improve_delta)) or (best == 0.0 and score > 0.05)
        if improved:
            fn = {"motorbike":"motorbike_best.jpg", "helmet":"helmet_best.jpg", "lp_img":"lp_best.jpg"}[kind]
            path = os.path.join(out_dir, fn)
            cv2.imwrite(path, crop)
            self.best_scores[kind] = score
            self.best_paths[kind] = path
            return path, score, True
        return self.best_paths.get(kind), score, False

# ---------- Track object ----------
@dataclass
class Track:
    track_id: int
    created_at: str
    start_frame: int
    last_seen_frame: int
    bbox: Tuple[int,int,int,int]
    misses: int = 0
    frames: int = 0
    helmet: HelmetEMA = field(default_factory=lambda: HelmetEMA())
    plate_fuser: PlateTextFuser = field(default_factory=lambda: PlateTextFuser())
    snapshots: SnapshotSelector = field(default_factory=lambda: SnapshotSelector())
    fullframe_best_score: float = float("-inf")
    session_dir: Optional[str] = None

# ---------- TrackManager ----------
class TrackManager:
    def __init__(self, output_root="data",
                 iou_thr=0.3, lost_frames=30,
                 vio_frames=15,
                 ema_alpha=0.7, yes_thr=0.6, no_thr=0.4,
                 entropy_lock=0.6, min_plate_repeats=3,
                 improve_delta=0.15):
        self.output_root = ensure_dir(output_root)
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.iou_thr = float(iou_thr)
        self.lost_frames = int(lost_frames)
        self.vio_frames = int(vio_frames)
        self.settings = {
            "ema_alpha": ema_alpha, "yes_thr": yes_thr, "no_thr": no_thr,
            "entropy_lock": entropy_lock, "min_plate_repeats": min_plate_repeats,
            "improve_delta": improve_delta,
        }
        self._seen_this_frame: Dict[int, bool] = {}

    # --- NEW: trong class TrackManager ---
    def summarize_track(self, t):
        """
        Trả về summary theo schema builders cần:
          - helmet_state / helmet_ema / frames_observed / track_id
          - plate_final / plate_conf (có thể rỗng & 0.0 nếu chưa lock)
          - best snapshots: motorbike / helmet / lp_img / full_frame (có thể None)
        """
        # Plate
        plate_text, plate_conf = "", 0.0
        if getattr(t, "plate_fuser", None) is not None:
            pf = t.plate_fuser
            if getattr(pf, "locked", False):
                plate_text = pf.final_text or ""
                plate_conf = float(getattr(pf, "final_conf", 0.0) or 0.0)
            else:
                # Chưa lock: lấy candidate làm tham khảo, confidence = 0.0
                plate_text = pf.candidate() or ""
                plate_conf = 0.0
        
        # Áp dụng post_format_plate để chuyển đổi ký tự thứ 3
        plate_text = post_format_plate(plate_text)

        # Best snapshots (có thể thiếu full_frame nếu bạn chưa lưu full-frame)
        best_paths = getattr(t, "snapshots", None)
        best = {"motorbike": None, "helmet": None, "lp_img": None, "full_frame": None}
        if best_paths and hasattr(best_paths, "best_paths"):
            best.update(best_paths.best_paths)

        return {
            "track_id": t.track_id,
            "frames_observed": t.frames,
            "helmet_ema": t.helmet.ema if getattr(t, "helmet", None) else 0.0,
            "helmet_state": t.helmet.state if getattr(t, "helmet", None) else "unknown",
            "plate_final": plate_text,
            "plate_conf": plate_conf,
            "best": best
        }

    # ---- lifecycle ----
    def _new_track(self, det_bbox, frame_idx):
        tid = self.next_id; self.next_id += 1
        t = Track(
            track_id=tid, created_at=now_iso(),
            start_frame=frame_idx, last_seen_frame=frame_idx,
            bbox=tuple(map(int, det_bbox)),
            helmet=HelmetEMA(self.settings["ema_alpha"], self.settings["yes_thr"], self.settings["no_thr"]),
            plate_fuser=PlateTextFuser(entropy_lock=self.settings["entropy_lock"], min_repeats=self.settings["min_plate_repeats"]),
            snapshots=SnapshotSelector(improvement_delta=self.settings["improve_delta"]),
        )
        sess = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{tid:04d}"
        t.session_dir = ensure_dir(os.path.join(self.output_root, "tracks", sess))
        self._write_track_meta(t)
        self.tracks[tid] = t
        return t

    def _write_track_meta(self, t: Track):
        # Áp dụng post_format_plate cho plate_candidate
        plate_candidate = post_format_plate(t.plate_fuser.candidate())
        
        meta = {
            "session_dir": t.session_dir,
            "track_id": t.track_id,
            "created_at": t.created_at,
            "last_seen_at": now_iso(),
            "frames_seen": t.frames,
            "helmet_ema": t.helmet.ema,
            "helmet_state": t.helmet.state,
            "plate_candidate": plate_candidate,
            "plate_entropy": t.plate_fuser.avg_entropy(),
            "best": {
                "motorbike": {"path": t.snapshots.best_paths["motorbike"], "score": t.snapshots.best_scores["motorbike"]},
                "helmet":    {"path": t.snapshots.best_paths["helmet"],    "score": t.snapshots.best_scores["helmet"]},
                "lp_img":    {"path": t.snapshots.best_paths["lp_img"],    "score": t.snapshots.best_scores["lp_img"]},
            }
        }
        path = os.path.join(t.session_dir, "meta.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # ---- association (greedy IOU) ----
    def associate_and_update(self, frame_bgr, dets_xyxy: List[Tuple[int,int,int,int]], frame_idx: int):
        self._seen_this_frame = {}
        unmatched = set(range(len(dets_xyxy)))
        assignments = []
        for tid, t in self.tracks.items():
            best_j, best_iou = -1, 0.0
            for j in unmatched:
                iouv = iou_xyxy(t.bbox, dets_xyxy[j])
                if iouv > best_iou:
                    best_iou, best_j = iouv, j
            if best_j >= 0 and best_iou >= self.iou_thr:
                t.bbox = tuple(map(int, dets_xyxy[best_j]))
                t.last_seen_frame = frame_idx
                t.misses = 0
                t.frames += 1
                self._seen_this_frame[tid] = True
                assignments.append((t, dets_xyxy[best_j]))
                unmatched.remove(best_j)
        for j in list(unmatched):
            t = self._new_track(dets_xyxy[j], frame_idx)
            t.frames = 1
            self._seen_this_frame[t.track_id] = True
            assignments.append((t, dets_xyxy[j]))
        lost = []
        for tid, t in list(self.tracks.items()):
            if not self._seen_this_frame.get(tid, False):
                t.misses += 1
                if t.misses > self.lost_frames:
                    lost.append(tid)
        return assignments, lost

    # ---- per-track updates from M2/OCR ----
    def update_helmet_for_track(self, t: Track, helmet_prob: float, conf: float = 1.0):
        t.helmet.update(helmet_prob, conf)

    def add_plate_reading(self, t: Track, text: str, conf_avg: float, conf_vec: Optional[List[float]]=None):
        if t.session_dir:
            p = os.path.join(t.session_dir, "lp_readings.jsonl")
            rec = {"ts": now_iso(), "text": text, "conf": conf_avg}
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False)+"\n")
        if not conf_vec or not hasattr(conf_vec, "__len__"):
            conf_vec = [conf_avg] * min(len(text), 12)
        t.plate_fuser.update(text, conf_vec)

    def _inside_fraction(self, box, W, H):
        # box = (x1,y1,x2,y2)
        if box is None:
            return 0.0
        x1, y1, x2, y2 = map(int, box)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return 0.0
        # giao với khung
        ix1, iy1 = max(0, x1), max(0, y1)
        ix2, iy2 = min(W, x2), min(H, y2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area = bw * bh
        return inter / max(area, 1)

    def _sharpness_varlap(self, img_roi):
        # độ nét đơn giản: phương sai Laplacian
        import cv2, numpy as np
        if img_roi is None or img_roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def consider_full_frame(self, t: Track, frame_bgr, mb_box, helmet_box=None, plate_box=None):
        """
        Cập nhật ảnh full-frame tốt nhất cho 1 track:
          - Ưu tiên hộp 'motorbike' (bắt buộc phải có).
          - Bảo đảm 3 bbox nằm trong khung: motorbike, helmet, plate (nếu có).
          - Chấm điểm theo: độ phủ trong khung, kích thước xe, độ nét, độ cân giữa.
          - Nếu điểm > best_score hiện có -> ghi vào tracks/<TRACK_ID>/frames/full_frame.jpg
        """
        import os, cv2, math
        if not t.session_dir or mb_box is None:
            return

        H, W = frame_bgr.shape[:2]

        # 1) Tính các tỉ lệ "nằm trong khung"
        vis_mb = self._inside_fraction(mb_box, W, H)
        vis_he = self._inside_fraction(helmet_box, W, H) if helmet_box else 1.0  # không có thì không phạt mạnh
        vis_lp = self._inside_fraction(plate_box, W, H) if plate_box else 1.0

        # Nếu xe bị cắt nhiều thì loại (ví dụ < 0.95)
        if vis_mb < 0.95:
            return

        # 2) Kích thước xe tương đối (tránh quá nhỏ)
        x1, y1, x2, y2 = map(int, mb_box)
        bw, bh = x2 - x1, y2 - y1
        size_rel = (bw * bh) / float(W * H)  # 0..1

        # 3) Độ cân giữa (xe gần giữa khung)
        cx_mb = (x1 + x2) / 2.0
        cy_mb = (y1 + y2) / 2.0
        dx = abs(cx_mb - W / 2.0) / (W / 2.0)
        dy = abs(cy_mb - H / 2.0) / (H / 2.0)
        center_score = 1.0 - min(1.0, math.hypot(dx, dy))  # 1.0 = rất giữa

        # 4) Độ nét (var Laplacian) trên ROI xe (clamp để ổn định)
        roi = frame_bgr[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        sharp = self._sharpness_varlap(roi)
        sharp_norm = min(sharp / 200.0, 1.0)  # scale thô; có thể tinh chỉnh theo video của bạn

        # 5) Hợp nhất điểm (trọng số có thể điều chỉnh)
        #   - bắt buộc thấy trọn xe; helmet/plate nếu có thì thưởng thêm
        vis_all = min(vis_mb, vis_he, vis_lp)
        score = (
                0.50 * vis_all +  # ưu tiên không bị cắt
                0.20 * min(size_rel / 0.25, 1.0) +  # xe không quá nhỏ
                0.20 * center_score +  # ở gần giữa khung
                0.10 * sharp_norm  # ảnh nét
        )

        # Nếu điểm tốt hơn trước -> ghi đè
        if score > getattr(t, "fullframe_best_score", float("-inf")):
            out_dir = os.path.join(t.session_dir, "")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "full_frame.jpg")
            cv2.imwrite(out_path, frame_bgr)
            t.fullframe_best_score = score
            # ghi lại để persistence copy vào evidence
            t.snapshots.best_paths["full_frame"] = out_path

    def offer_snapshot(self, t: Track, frame_bgr, bbox, kind: str, parent_bbox=None):
        if not t.session_dir: return
        path, score, changed = t.snapshots.maybe_update_and_save(t.session_dir, frame_bgr, bbox, kind, parent_bbox)
        if changed:
            # Khi snapshot (helmet / lp_img / motorbike) được cải thiện,
            # lưu luôn full_frame tương ứng để đảm bảo khớp ngữ cảnh.
            try:
                out_path = os.path.join(t.session_dir, "full_frame.jpg")
                cv2.imwrite(out_path, frame_bgr)
                t.snapshots.best_paths["full_frame"] = out_path
            except Exception:
                pass
            self._write_track_meta(t)

    def offer_full_frame(self, t: Track, frame_bgr):
        """
        Lưu toàn bộ khung hình video vào thư mục data/full_frame/<track_id>.jpg
        (mỗi track chỉ lưu 1 lần).
        """
        if not t.session_dir:
            return
        if "full_frame" in t.snapshots.best_paths:
            return

        # Lấy thư mục gốc "data/" từ session_dir (ví dụ: data/tracks/...)
        root_dir = os.path.dirname(os.path.dirname(t.session_dir))  # -> ./data
        out_dir = os.path.join(root_dir, "full_frame")
        os.makedirs(out_dir, exist_ok=True)

        # Track dùng thuộc tính track_id (không phải id)
        track_name = getattr(t, "track_id", None)
        if track_name is None:
            # fallback an toàn nếu cấu trúc khác
            track_name = "track"

        out_path = os.path.join(out_dir, f"{track_name}.jpg")
        cv2.imwrite(out_path, frame_bgr)

        # Ghi lại đường dẫn để persistence copy vào evidence khi cần
        t.snapshots.best_paths["full_frame"] = out_path

    # ---- decisions ----
    def should_raise_violation(self, t: Track) -> bool:
        return (t.helmet.state == "no_helmet")# and
        #        t.frames >= self.vio_frames and
        #        t.plate_fuser.locked)

    def should_record_observation(self, t: Track) -> bool:
        return (t.helmet.state == "helmet" and
                t.frames >= max(8, self.vio_frames//2) and
                t.plate_fuser.locked)

    def finalize_track(self, tid: int):
        t = self.tracks.pop(tid, None)
        if not t: return None
        self._write_track_meta(t)
        # Lấy plate text và áp dụng post_format_plate
        plate_text = t.plate_fuser.final_text if t.plate_fuser.locked else t.plate_fuser.candidate()
        plate_text = post_format_plate(plate_text)
        
        summary = {
            "track_id": t.track_id,
            "created_at": t.created_at,
            "last_seen_at": now_iso(),
            "frames_observed": t.frames,
            "helmet_ema": t.helmet.ema,
            "helmet_state": t.helmet.state,
            "plate_final": plate_text,
            "plate_conf": t.plate_fuser.final_conf if t.plate_fuser.locked else 0.0,
            "session_dir": t.session_dir,
            "best": t.snapshots.best_paths
        }
        summary["decision"] = ("violation" if self.should_raise_violation(t)
                               else ("observation" if self.should_record_observation(t)
                                     else "none"))
        return summary
