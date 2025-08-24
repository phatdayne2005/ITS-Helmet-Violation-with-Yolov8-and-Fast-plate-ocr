# main.py
# ================================
# Hệ thống Giám sát ATGT - Phát hiện vi phạm không đội nón bảo hiểm
# 
# Kiến trúc:
# - Stage 1: Phát hiện xe máy (YOLO)
# - Stage 2: Phát hiện nón bảo hiểm và biển số xe (YOLO)
# - OCR: Nhận dạng biển số xe (fast-plate-ocr)
# - Tracking: Theo dõi đối tượng qua các frame
# - Persistence: Lưu trữ dữ liệu vi phạm và quan sát
# - Email: Gửi cảnh báo vi phạm
# ================================
import os
import time
import threading
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch
try:
    # Fix PyTorch >= 2.6 khi load model Ultralytics
    from ultralytics.nn import tasks as yolo_tasks
    torch.serialization.add_safe_globals([yolo_tasks.DetectionModel])
except Exception:
    pass
from ultralytics import YOLO

# OCR bằng fast-plate-ocr
try:
    from fast_plate_ocr import LicensePlateRecognizer
except Exception:
    LicensePlateRecognizer = None

# Tracking + JSON
from module.traffic_core import TrackManager
from module.persistence import DataPaths, JsonStore, IncidentBuilder, ObservationBuilder

# Email integration
try:
    from module.email import send_helmet_warning_email
except ImportError:
    send_helmet_warning_email = None

# ============ Helpers ============
def draw_ocr_label(img, box, text, conf=None, color=(255, 0, 0), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if not text:
        return
    label = f"{text}" if conf is None else f"{text} {conf:.2f}"
    h = max(1, y2 - y1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, min(1.2, h / 60.0))
    (tw, th), bl = cv2.getTextSize(label, font, fs, 2)
    lx1, ly1 = x1, y1 - th - 6
    if ly1 < 0:
        ly1 = y1 + 6
    lx2, ly2 = lx1 + tw + 10, ly1 + th + bl + 6
    lx1 = max(0, lx1); ly1 = max(0, ly1)
    lx2 = min(img.shape[1]-1, lx2); ly2 = min(img.shape[0]-1, ly2)
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.putText(img, label, (lx1 + 5, ly2 - bl - 3), font, fs, (255, 255, 255), 2, cv2.LINE_AA)

def is_lp_label(name: str) -> bool:
    n = str(name).lower().replace("_", " ").strip()
    return ("plate" in n) or (n in {"lp", "license plate", "number plate", "bien so", "bienso"})

def normalize_char(lbl: str) -> str:
    if not lbl:
        return ""
    ch = str(lbl)[0].upper()
    if ("0" <= ch <= "9") or ("A" <= ch <= "Z"):
        return ch
    return ""

def post_format_plate(s: str) -> str:
    s = "".join(ch for ch in s if ch.isalnum()).upper()
    if not s:
        return ""
    
    # Xử lý theo quy tắc biển số Việt Nam
    # 1. Chữ "O" luôn chuyển thành "0" (biển số VN không có chữ O)
    s = s.replace("O", "0")
    
    return s

# ============ GUI ============
class YoloTesterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Giám sát ATGT - Phát hiện vi phạm không đội nón bảo hiểm")
        self.root.geometry("1200x720")

        # State
        self.model = None             # M1: motorbike
        self.model2 = None            # M2: helmet/LP
        self.fpocr = None             # fast-plate-ocr
        self.ocr_model = None         # fallback YOLOv8 char-detector (không khuyến nghị)

        self.running = False
        self.cap = None
        self.display_lock = threading.Lock()
        self.frame_thread = None

        # Params
        self.source_type = tk.StringVar(value="image")
        self.weights_path = tk.StringVar(value="model/motorbike.pt")
        self.weights2_path = tk.StringVar(value="model/helmet_lp.pt")
        self.ocr_weights = tk.StringVar(value="cct-xs-v1-global-model")  # tên HUB hoặc .onnx

        self.source_path = tk.StringVar()
        self.device_choice = tk.StringVar(value="")
        self.imgsz = tk.IntVar(value=640)
        self.conf = tk.DoubleVar(value=0.30)      # conf M1
        self.conf2 = tk.DoubleVar(value=0.35)     # conf M2

        # fallback params (YOLOv8 char-detector)
        self.ocr_conf = tk.DoubleVar(value=0.30)
        self.ocr_cls_offset = tk.IntVar(value=1)

        self.class_filter = tk.StringVar(value="")
        self.enable_model2 = tk.BooleanVar(value=True)
        self.enable_ocr = tk.BooleanVar(value=True)

        # === Tracking + JSON persistence ===
        self.frame_idx = 0
        self.paths = DataPaths(root="data")       # auto tạo thư mục
        self.store = JsonStore(self.paths)
        self.tm = TrackManager(
            output_root="data",
            iou_thr=0.3, lost_frames=30, vio_frames=15,
            ema_alpha=0.7, yes_thr=0.6, no_thr=0.4,
            entropy_lock=0.6, min_plate_repeats=3,
            improve_delta=0.15
        )
        self.inc_builder = IncidentBuilder(self.paths, self.store, camera_id="CAM_01", location="Gate A")
        self.obs_builder = ObservationBuilder(self.paths)

        # === Email configuration ===
        # Cấu hình email mặc định - có thể thay đổi trực tiếp ở đây
        self.email_config = {
            "smtp_user": "",  # Thay đổi email của bạn
            "smtp_password": "",    # Thay đổi App Password của bạn
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_name": "Hệ thống Cảnh báo ATGT"
        }

        # UI
        self._build_ui()
        
        # Auto load models on startup
        self.auto_load_models()

    def auto_load_models(self):
        """
        Tự động load các model khi khởi động ứng dụng
        """
        try:
            # Load model motorbike
            if Path(self.weights_path.get()).exists():
                self.load_weights()
            else:
                self.log("Không tìm thấy model motorbike.pt, vui lòng chọn file model.")
            
            # Load model helmet/lp
            if Path(self.weights2_path.get()).exists():
                self.load_weights2()
            else:
                self.log("Không tìm thấy model helmet_lp.pt, vui lòng chọn file model.")
            
            # Load OCR model
            self.load_ocr()
            
        except Exception as e:
            self.log(f"Lỗi khi tự động load models: {e}")

    # --- NEW: trong class YoloTesterGUI ---
    def flush_all_tracks(self):
        """
        Kết luận tất cả các track đang còn sống:
          - no_helmet -> Incident
          - còn lại   -> Observation
        Tránh tạo trùng bằng cờ _finalized trên từng track.
        """
        try:
            count_inc, count_obs = 0, 0
            for t in list(self.tm.tracks.values()):
                if getattr(t, "_finalized", False):
                    continue  # đã kết luận trước đó

                summary = self.tm.summarize_track(t)
                if not summary:
                    continue

                if summary.get("helmet_state") == "no_helmet":
                    # 1) Tạo INCIDENT
                    incident = self.inc_builder.build_from_summary(summary)  # tạo thư mục incidents + index
                    count_inc += 1

                    # 2) Gửi EMAIL ngay sau khi tạo incident (chạy ở thread riêng để không chặn UI)
                    try:
                        threading.Thread(
                            target=self.send_violation_email,
                            args=(incident,),
                            daemon=True
                        ).start()
                    except Exception as e:
                        self.log(f"Email send error for {incident.get('id', '?')}: {e}")
                else:
                    # Không vi phạm -> Observation
                    self.obs_builder.build_from_summary(summary)
                    count_obs += 1

                setattr(t, "_finalized", True)

            self.log(f"Flushed tracks -> incidents: {count_inc}, observations: {count_obs}")
        except Exception as e:
            self.log(f"[WARN] Flush tracks error: {e}")

    # ---------- UI ----------
    def _build_ui(self):
        left = ttk.Frame(self.root, padding=10)
        left.pack(side="left", fill="y")

        # Weights M1
        ttk.Label(left, text="M1 Weights (.pt) - Motorbike:").pack(anchor="w")
        w1 = ttk.Frame(left); w1.pack(fill="x", pady=4)
        ttk.Entry(w1, textvariable=self.weights_path, width=36).pack(side="left", fill="x", expand=True)
        ttk.Button(w1, text="Chọn...", command=self.browse_weights).pack(side="left", padx=5)

        # Weights M2
        ttk.Label(left, text="M2 Weights (.pt) - Helmet/LP:").pack(anchor="w", pady=(6, 0))
        w2 = ttk.Frame(left); w2.pack(fill="x", pady=4)
        ttk.Entry(w2, textvariable=self.weights2_path, width=36).pack(side="left", fill="x", expand=True)
        ttk.Button(w2, text="Chọn...", command=self.browse_weights2).pack(side="left", padx=5)

        # Model 2 options
        m2f = ttk.LabelFrame(left, text="Model 2 Options", padding=10)
        m2f.pack(fill="x", pady=6)
        r = ttk.Frame(m2f); r.pack(fill="x", pady=2)
        ttk.Label(r, text="conf2:").pack(side="left")
        ttk.Entry(r, textvariable=self.conf2, width=8).pack(side="left", padx=6)
        ttk.Checkbutton(m2f, text="Bật Model 2", variable=self.enable_model2).pack(anchor="w", pady=(6, 0))

        # OCR (fast-plate-ocr)
        ocrf = ttk.LabelFrame(left, text="OCR (fast-plate-ocr)", padding=10)
        ocrf.pack(fill="x", pady=6)
        ro = ttk.Frame(ocrf); ro.pack(fill="x", pady=2)
        ttk.Label(ro, text="FPOCR model/onnx:").pack(side="left")
        ttk.Entry(ro, textvariable=self.ocr_weights, width=28).pack(side="left", padx=6)
        ttk.Button(ro, text="Chọn .onnx", command=self.browse_ocr_weights).pack(side="left")
        ttk.Label(ocrf, text="Gợi ý: cct-xs-v1-global-model (mặc định)").pack(anchor="w", pady=(6, 0))
        ttk.Checkbutton(ocrf, text="Bật OCR", variable=self.enable_ocr).pack(anchor="w", pady=(6, 0))

        # Source
        ttk.Label(left, text="Nguồn dữ liệu:").pack(anchor="w", pady=(8, 0))
        src_frame = ttk.Frame(left); src_frame.pack(fill="x", pady=4)
        for txt, val in [("Ảnh", "image"), ("Thư mục", "folder"), ("Video", "video"), ("Webcam", "webcam")]:
            ttk.Radiobutton(src_frame, text=txt, value=val, variable=self.source_type,
                            command=self._on_source_type_change).pack(side="left", padx=2)

        self.src_label = ttk.Label(left, text="Đường dẫn nguồn:")
        self.src_label.pack(anchor="w")
        s_frame = ttk.Frame(left); s_frame.pack(fill="x", pady=4)
        ttk.Entry(s_frame, textvariable=self.source_path, width=36).pack(side="left", fill="x", expand=True)
        self.src_btn = ttk.Button(s_frame, text="Chọn...", command=self.browse_source)
        self.src_btn.pack(side="left", padx=5)

        # Device & params
        ttk.Label(left, text="Thiết bị (CPU/GPU):").pack(anchor="w", pady=(8, 0))
        d = ttk.Frame(left); d.pack(fill="x", pady=4)
        ttk.Combobox(d, textvariable=self.device_choice, values=["", "cpu", "cuda:0"], width=10).pack(side="left")
        ttk.Label(d, text="(để trống = auto)").pack(side="left", padx=6)

        pf = ttk.LabelFrame(left, text="Tham số suy luận", padding=10); pf.pack(fill="x", pady=6)
        r = ttk.Frame(pf); r.pack(fill="x", pady=2)
        ttk.Label(r, text="imgsz:").pack(side="left")
        ttk.Entry(r, textvariable=self.imgsz, width=8).pack(side="left", padx=6)
        r2 = ttk.Frame(pf); r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="conf1:").pack(side="left")
        ttk.Entry(r2, textvariable=self.conf, width=8).pack(side="left", padx=6)
        r3 = ttk.Frame(pf); r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Lọc class (M1):").pack(side="left")
        ttk.Entry(r3, textvariable=self.class_filter, width=18).pack(side="left", padx=6)

        # Actions
        af = ttk.Frame(left); af.pack(fill="x", pady=8)
        ttk.Button(af, text="Load M1", command=self.load_weights).pack(side="left", padx=2)
        ttk.Button(af, text="Load M2", command=self.load_weights2).pack(side="left", padx=2)
        ttk.Button(af, text="Load OCR", command=self.load_ocr).pack(side="left", padx=2)
        ttk.Button(af, text="Start", command=self.start).pack(side="left", padx=2)
        ttk.Button(af, text="Stop", command=self.stop).pack(side="left", padx=2)
        ttk.Button(af, text="Lưu khung hiện tại", command=self.save_current_frame).pack(side="left", padx=2)

        # Email Status
        emailf = ttk.LabelFrame(left, text="Trạng thái Email", padding=10)
        emailf.pack(fill="x", pady=6)
        ttk.Label(emailf, text="✅ Email cảnh báo đã được bật", 
                  foreground="#008000", font=("", 10, "bold")).pack(anchor="w")
        ttk.Label(emailf, text="Email sẽ tự động gửi khi phát hiện vi phạm", 
                  foreground="#666", font=("", 9)).pack(anchor="w", pady=(4, 0))

        # Status
        self.status_var = tk.StringVar(value="Chưa tải weights.")
        ttk.Label(left, textvariable=self.status_var, foreground="#555").pack(anchor="w", pady=6)

        # Right display
        right = ttk.Frame(self.root, padding=10); right.pack(side="right", fill="both", expand=True)
        self.canvas = tk.Canvas(right, bg="black"); self.canvas.pack(fill="both", expand=True)
        self.tkimg_ref = None

        self._on_source_type_change()

    def log(self, text):
        self.status_var.set(text)
        try: self.root.update_idletasks()
        except Exception: pass

    # ---------- Browsers ----------
    def browse_weights(self):
        p = filedialog.askopenfilename(title="Chọn file weights .pt (M1)", filetypes=[("PyTorch Weights", "*.pt"), ("All files", "*.*")])
        if p: self.weights_path.set(p)

    def browse_weights2(self):
        p = filedialog.askopenfilename(title="Chọn file weights .pt (M2)", filetypes=[("PyTorch Weights", "*.pt"), ("All files", "*.*")])
        if p: self.weights2_path.set(p)

    def browse_ocr_weights(self):
        p = filedialog.askopenfilename(title="Chọn file model .onnx (FPOCR)", filetypes=[("ONNX", "*.onnx"), ("All files", "*.*")])
        if p: self.ocr_weights.set(p)

    def _on_source_type_change(self):
        t = self.source_type.get()
        if t == "webcam":
            self.src_label.config(text="Webcam index:")
            self.source_path.set("0")
            self.src_btn.config(state="disabled")
        elif t == "folder":
            self.src_label.config(text="Thư mục ảnh:")
            self.src_btn.config(state="normal")
        elif t == "video":
            self.src_label.config(text="File video:")
            self.src_btn.config(state="normal")
        else:
            self.src_label.config(text="File ảnh:")
            self.src_btn.config(state="normal")

    def browse_source(self):
        t = self.source_type.get()
        if t == "folder":
            p = filedialog.askdirectory(title="Chọn thư mục ảnh")
        elif t == "video":
            p = filedialog.askopenfilename(title="Chọn video", filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.flv"), ("All files", "*.*")])
        else:
            p = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp"), ("All files", "*.*")])
        if p: self.source_path.set(p)

    # ---------- Load models ----------
    def load_weights(self):
        w = self.weights_path.get().strip()
        if not w:
            messagebox.showwarning("Thiếu weights", "Hãy chọn file .pt (M1)."); return
        if not Path(w).exists():
            messagebox.showerror("Không tìm thấy", f"Không tìm thấy file: {w}"); return
        try:
            self.model = YOLO(w)
            if self.device_choice.get():
                self.model.to(self.device_choice.get())
            self.log(f"Đã load M1: {Path(w).name}")
        except Exception as e:
            self.model = None
            messagebox.showerror("Lỗi load M1", str(e))
            self.log("Load M1 thất bại.")

    def load_weights2(self):
        w = self.weights2_path.get().strip()
        if not w:
            messagebox.showwarning("Thiếu weights", "Hãy chọn file .pt (M2)."); return
        if not Path(w).exists():
            messagebox.showerror("Không tìm thấy", f"Không tìm thấy file: {w}"); return
        try:
            self.model2 = YOLO(w)
            if self.device_choice.get():
                self.model2.to(self.device_choice.get())
            self.log(f"Đã load M2: {Path(w).name}")
        except Exception as e:
            self.model2 = None
            messagebox.showerror("Lỗi load M2", str(e))
            self.log("Load M2 thất bại.")

    def load_ocr(self):
        model_id_or_path = self.ocr_weights.get().strip() or "cct-xs-v1-global-model"
        if LicensePlateRecognizer is None:
            messagebox.showwarning(
                "Chưa có fast-plate-ocr",
                "Bạn chưa cài fast-plate-ocr. Chạy:\n\npip install \"fast-plate-ocr[onnx]\""
            )
            self.fpocr = None
        else:
            try:
                dev_raw = self.device_choice.get().strip().lower()
                device = "cuda" if dev_raw.startswith("cuda") else ("cpu" if dev_raw == "cpu" else None)
                if device:
                    self.fpocr = LicensePlateRecognizer(model_id_or_path, device=device)
                else:
                    self.fpocr = LicensePlateRecognizer(model_id_or_path)
                self.log(f"Đã load FPOCR: {model_id_or_path}")
                return
            except Exception as e:
                self.fpocr = None
                messagebox.showerror("Lỗi load FPOCR", str(e))

        # Fallback YOLOv8 char-detector (chỉ khi người dùng chọn .pt)
        w = self.ocr_weights.get().strip()
        if w and Path(w).exists() and w.lower().endswith(".pt"):
            try:
                self.ocr_model = YOLO(w)
                if self.device_choice.get():
                    self.ocr_model.to(self.device_choice.get())
                self.log(f"(Fallback) Đã load OCR YOLOv8: {Path(w).name}")
            except Exception as e:
                self.ocr_model = None
                messagebox.showerror("Lỗi load OCR (YOLOv8)", str(e))
                self.log("Load OCR (YOLOv8) thất bại.")
        else:
            self.ocr_model = None

    # ---------- Email functions ----------
    def send_violation_email(self, incident: dict):
        """
        Gửi email cảnh báo vi phạm không đội nón bảo hiểm
        """
        if send_helmet_warning_email is None:
            self.log("Email: Chưa import được email_test.py")
            return
        
        # Kiểm tra cấu hình email
        smtp_user = self.email_config["smtp_user"]
        smtp_password = self.email_config["smtp_password"]
        if not smtp_user or not smtp_password:
            self.log("Email: Thiếu cấu hình SMTP")
            return
        
        # Lấy thông tin chủ xe
        plate = incident.get("plate", {}).get("text", "")
        if not plate:
            self.log("Email: Không có biển số xe")
            return
        
        owner_info = self.store.find_owner(plate)
        if not owner_info.get("found") or not owner_info.get("email"):
            self.log(f"Email: Không tìm thấy thông tin chủ xe {plate}")
            return
        
        # Chuẩn bị ảnh evidence
        evidence_images = []
        incident_dir = os.path.join(self.paths.incidents_dir, incident["id"])
        
        # Thêm ảnh full frame (bối cảnh tổng thể)
        full_frame_path = os.path.join(incident_dir, "evidence", "full_frame.jpg")
        if os.path.exists(full_frame_path):
            evidence_images.append(full_frame_path)
        
        # Thêm ảnh helmet (vi phạm)
        helmet_path = os.path.join(incident_dir, "evidence", "helmet.jpg")
        if os.path.exists(helmet_path):
            evidence_images.append(helmet_path)
        
        # Thêm ảnh biển số
        plate_path = os.path.join(incident_dir, "evidence", "plate.jpg")
        if os.path.exists(plate_path):
            evidence_images.append(plate_path)
        
        try:
            # Gửi email
            send_helmet_warning_email(
                to_email=owner_info["email"],
                images=evidence_images,
                smtp_user=smtp_user,
                smtp_password=smtp_password,
                smtp_server=self.email_config["smtp_server"],
                smtp_port=self.email_config["smtp_port"],
                sender_name=self.email_config["sender_name"],
                recipient_name=owner_info["name"],
                subject=f"Cảnh báo: Không đội nón bảo hiểm - Biển số {plate}"
            )
            
            # Cập nhật trạng thái email
            sent_at = time.strftime("%Y-%m-%d %H:%M:%S")
            incident["email"]["status"] = "sent"
            incident["email"]["sent_at"] = sent_at
            
            # Cập nhật incident.json
            incident_path = os.path.join(incident_dir, "incident.json")
            import json
            try:
                with open(incident_path, 'r', encoding='utf-8') as f:
                    incident_data = json.load(f)
                incident_data["email"]["status"] = "sent"
                incident_data["email"]["sent_at"] = sent_at
                with open(incident_path, 'w', encoding='utf-8') as f:
                    json.dump(incident_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.log(f"Email: Lỗi cập nhật incident.json: {e}")
            
            # Cập nhật index
            self.update_incident_index(incident["id"], "sent", sent_at)
            
            self.log(f"Email: Đã gửi cảnh báo đến {owner_info['name']} ({owner_info['email']})")
            
        except Exception as e:
            self.log(f"Email: Lỗi gửi email: {e}")
            # Cập nhật trạng thái lỗi
            error_msg = str(e)
            incident["email"]["status"] = "failed"
            incident["email"]["error"] = error_msg
            
            try:
                incident_path = os.path.join(incident_dir, "incident.json")
                import json
                with open(incident_path, 'r', encoding='utf-8') as f:
                    incident_data = json.load(f)
                incident_data["email"]["status"] = "failed"
                incident_data["email"]["error"] = error_msg
                with open(incident_path, 'w', encoding='utf-8') as f:
                    json.dump(incident_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            
            # Cập nhật index
            self.update_incident_index(incident["id"], "failed", error=error_msg)

    def update_incident_index(self, incident_id: str, email_status: str, sent_at: str = "", error: str = ""):
        """
        Cập nhật thông tin email trong incidents index
        """
        try:
            import json
            index_path = self.paths.incidents_index
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            # Tìm và cập nhật incident
            for item in index:
                if item["id"] == incident_id:
                    item["email_status"] = email_status
                    if sent_at:
                        item["email_sent_at"] = sent_at
                    if error:
                        item["email_error"] = error
                    break
            
            # Lưu lại index
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.log(f"Email: Lỗi cập nhật index: {e}")



    # ---------- OCR bằng fast-plate-ocr ----------
    def _ocr_fast_plate_ocr(self, lp_bgr):
        if self.fpocr is None or lp_bgr is None or lp_bgr.size == 0:
            return "", 0.0
        try:
            res = self.fpocr.run(lp_bgr, return_confidence=True)
        except Exception:
            return "", 0.0
        plates, confs = None, None
        if isinstance(res, tuple) and len(res) == 2:
            plates, confs = res
        else:
            plates = res
        if isinstance(plates, (list, tuple)):
            text_raw = str(plates[0]) if len(plates) else ""
        else:
            text_raw = str(plates or "")
        conf_avg = 0.0
        if confs is not None:
            try:
                c0 = confs[0] if isinstance(confs, (list, tuple)) else confs
                if hasattr(c0, "__iter__"):
                    conf_avg = float(np.mean(list(c0)))
                else:
                    conf_avg = float(c0)
            except Exception:
                conf_avg = 0.0
        text_raw = text_raw.replace("_", "").strip()
        plate = post_format_plate(text_raw)
        return plate, conf_avg

    # ---------- OCR bằng YOLOv8 (fallback) ----------
    def _ocr_yolov8_chars(self, lp_bgr):
        if self.ocr_model is None:
            return "", 0.0
        res = self.ocr_model.predict(source=lp_bgr, imgsz=320, conf=float(self.ocr_conf.get()), verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return "", 0.0
        raw_names = getattr(res, "names", None) or getattr(self.ocr_model, "names", {}) or {}
        names = {i: raw_names[i] for i in range(len(raw_names))} if isinstance(raw_names, list) else dict(raw_names)
        digit_ids  = [i for i, lb in names.items() if str(lb) and str(lb)[0].isdigit()]
        letter_ids = [i for i, lb in names.items() if str(lb) and str(lb)[0].isalpha()]
        digit_ids.sort(); letter_ids.sort()
        offset = int(self.ocr_cls_offset.get())
        def map_with_offset(cid: int) -> str:
            base = names.get(cid, str(cid))
            if cid in digit_ids and len(digit_ids) > 0:
                pos = digit_ids.index(cid); nid = digit_ids[(pos + offset) % len(digit_ids)]
                return normalize_char(names.get(nid, base))
            if cid in letter_ids and len(letter_ids) > 0:
                pos = letter_ids.index(cid); nid = letter_ids[(pos + offset) % len(letter_ids)]
                return normalize_char(names.get(nid, base))
            return normalize_char(base)
        H, W = lp_bgr.shape[:2]
        chars = []
        for k in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[k].item()) if res.boxes.cls is not None else -1
            ch = map_with_offset(cls_id)
            if not ch: continue
            conf = float(res.boxes.conf[k].item()) if res.boxes.conf is not None else 0.0
            x1, y1, x2, y2 = map(float, res.boxes.xyxy[k].tolist())
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            chars.append((ch, cx, cy, conf))
        if not chars:
            return "", 0.0
        ys = np.array([c[2] for c in chars], dtype=float)
        two_lines = (ys.max() - ys.min()) > 0.35 * H
        if two_lines:
            y_mid = float(np.median(ys))
            top = [c for c in chars if c[2] <= y_mid]
            bot = [c for c in chars if c[2] >  y_mid]
            top.sort(key=lambda t: t[1]); bot.sort(key=lambda t: t[1])
            raw_text = "".join([c[0] for c in top]) + "".join([c[0] for c in bot])
            conf_avg = float(np.mean([c[3] for c in chars]))
        else:
            chars.sort(key=lambda t: t[1])
            raw_text = "".join([c[0] for c in chars])
            conf_avg = float(np.mean([c[3] for c in chars]))
        plate = post_format_plate(raw_text)
        return plate, conf_avg

    # ---------- Run ----------
    def start(self):
        if self.model is None:
            messagebox.showwarning("Chưa có model", "Hãy load weights M1 trước."); return
        if self.running:
            return
        t = self.source_type.get()
        src = self.source_path.get().strip()
        self.running = True

        if t == "webcam":
            try: index = int(src)
            except Exception: index = 0
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                self.running = False; messagebox.showerror("Webcam", f"Không mở được webcam index {index}"); return
            self.frame_thread = threading.Thread(target=self._loop_video_capture, daemon=True); self.frame_thread.start()

        elif t == "video":
            if not Path(src).exists():
                self.running = False; messagebox.showerror("Video", f"Không tìm thấy: {src}"); return
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                self.running = False; messagebox.showerror("Video", f"Không mở được: {src}"); return
            self.frame_thread = threading.Thread(target=self._loop_video_capture, daemon=True); self.frame_thread.start()

        elif t == "folder":
            folder = Path(src)
            if not folder.exists():
                self.running = False; messagebox.showerror("Thư mục", f"Không tìm thấy: {src}"); return
            self.frame_thread = threading.Thread(target=self._run_folder_images, args=(folder,), daemon=True)
            self.frame_thread.start()

        else:  # image
            if not Path(src).exists():
                self.running = False; messagebox.showerror("Ảnh", f"Không tìm thấy: {src}"); return
            self.frame_thread = threading.Thread(target=self._run_single_image, args=(src,), daemon=True)
            self.frame_thread.start()

    def stop(self):
        self.running = False
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
            # --- NEW: chốt tất cả track còn lại
        self.flush_all_tracks()
        self.log("Đã dừng.")

    # ---------- Core inference ----------
    def _predict_draw(self, frame_bgr):
        t0 = time.time()
        imgsz = int(self.imgsz.get())
        conf1 = float(self.conf.get())
        cls_filter = None
        if self.class_filter.get().strip():
            try:
                cls_filter = [int(x) for x in self.class_filter.get().strip().split()]
            except Exception:
                cls_filter = None

        # Model 1
        res1 = self.model.predict(source=frame_bgr, imgsz=imgsz, conf=conf1, classes=cls_filter, verbose=False)[0]
        annotated = frame_bgr.copy()

        # --- B1: ghép vào TrackManager ---
        m1_boxes = res1.boxes
        dets = []
        if m1_boxes is not None and len(m1_boxes) > 0:
            for i in range(len(m1_boxes)):
                x1, y1, x2, y2 = m1_boxes.xyxy[i].tolist()
                dets.append((int(x1), int(y1), int(x2), int(y2)))
        assignments, lost_ids = self.tm.associate_and_update(frame_bgr, dets, self.frame_idx)

        H, W = frame_bgr.shape[:2]

        # --- Duyệt từng track đã gán để chạy M2 + OCR + snapshot ---
        for t, det_bbox in assignments:
            x1i, y1i, x2i, y2i = map(int, det_bbox)
            helmet_box = None
            lp_box = None

            # vẽ motorbike + Track ID
            cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (0, 255, 255), 2)
            cv2.putText(annotated, f"ID {t.track_id}  motorbike",
                        (x1i, max(0, y1i - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # snapshot motorbike
            self.tm.offer_snapshot(t, frame_bgr, (x1i,y1i,x2i,y2i), "motorbike", parent_bbox=None)

            # Bỏ lưu full_frame tự động mỗi frame để tránh lệch ngữ cảnh.

            # Model 2 trong crop motorbike
            if self.enable_model2.get() and (self.model2 is not None):
                x1c, y1c, x2c, y2c = max(0, x1i), max(0, y1i), min(W-1, x2i), min(H-1, y2i)
                if x2c > x1c and y2c > y1c:
                    crop = frame_bgr[y1c:y2c, x1c:x2c]
                    res2 = self.model2.predict(source=crop, imgsz=imgsz, conf=float(self.conf2.get()), verbose=False)[0]
                    names2 = getattr(self.model2, "names", {})

                    if res2.boxes is not None and len(res2.boxes) > 0:
                        for j in range(len(res2.boxes)):
                            bx1, by1, bx2, by2 = res2.boxes.xyxy[j].tolist()
                            bx1i, by1i, bx2i, by2i = map(int, [bx1, by1, bx2, by2])
                            cls_id = int(res2.boxes.cls[j].item()) if res2.boxes.cls is not None else -1
                            conf_m2 = float(res2.boxes.conf[j].item()) if res2.boxes.conf is not None else 0.0
                            cls_name = names2.get(cls_id, str(cls_id)).strip()

                            gx1, gy1 = x1c + bx1i, y1c + by1i
                            gx2, gy2 = x1c + bx2i, y1c + by2i
                            gx1, gy1, gx2, gy2 = max(0,gx1), max(0,gy1), min(W-1,gx2), min(H-1,gy2)

                            low = cls_name.lower()
                            # Helmet / No-helmet
                            if "helmet" in low:
                                has_helmet = ("no" not in low)
                                # Suy ra prob từ nhãn lớp: helmet -> 1, no-helmet -> 0
                                prob = 1.0 if has_helmet else 0.0
                                # Dùng độ tự tin của model2 (conf_m2) làm trọng số EMA
                                self.tm.update_helmet_for_track(t, prob, conf_m2)
                                color = (0,255,0) if has_helmet else (0,0,255)
                                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), color, 2)
                                cv2.putText(annotated, f"{cls_name} {conf_m2:.2f}", (gx1, max(0, gy1 - 6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                                # snapshot helmet
                                self.tm.offer_snapshot(t, frame_bgr, (gx1,gy1,gx2,gy2), "helmet",
                                                       parent_bbox=(x1i,y1i,x2i,y2i))

                                # NEW: lưu lại bbox global của helmet cho frame hiện tại
                                helmet_box = (gx1, gy1, gx2, gy2)

                            # License plate -> OCR + snapshot
                            if is_lp_label(low):
                                if gx2 > gx1 and gy2 > gy1:
                                    lp_crop = frame_bgr[gy1:gy2, gx1:gx2]
                                    plate_text, conf_ocr = "", 0.0
                                    if self.enable_ocr.get() and (self.fpocr is not None) and lp_crop.size > 0:
                                        plate_text, conf_ocr = self._ocr_fast_plate_ocr(lp_crop)
                                    elif self.enable_ocr.get() and (self.ocr_model is not None) and lp_crop.size > 0:
                                        plate_text, conf_ocr = self._ocr_yolov8_chars(lp_crop)
                                    if plate_text:
                                        self.tm.add_plate_reading(t, plate_text, conf_ocr, conf_vec=None)
                                        self.tm.offer_snapshot(t, frame_bgr, (gx1,gy1,gx2,gy2), "lp_img",
                                                               parent_bbox=(x1i,y1i,x2i,y2i))
                                        txt = t.plate_fuser.final_text if t.plate_fuser.locked else t.plate_fuser.candidate()
                                        # Áp dụng post_format_plate để chuyển đổi ký tự thứ 3
                                        txt_formatted = post_format_plate(txt)
                                        draw_ocr_label(annotated, (gx1, gy1, gx2, gy2), txt_formatted, conf=None, color=(255,0,0), thickness=2)

                                        # NEW: lưu bbox global của biển số cho frame hiện tại
                                        lp_box = (gx1, gy1, gx2, gy2)
                                    else:
                                        cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (255,0,0), 2)

            # Ghi trạng thái track dưới bbox xe
            info = []
            info.append(f"helmet:{t.helmet.state}({t.helmet.ema:.2f})")
            cand = t.plate_fuser.final_text if t.plate_fuser.locked else t.plate_fuser.candidate()
            if cand: 
                # Áp dụng post_format_plate để chuyển đổi ký tự thứ 3
                cand_formatted = post_format_plate(cand)
                info.append(f"LP:{cand_formatted}")
            cv2.putText(annotated, " | ".join(info), (x1i, min(H-5, y2i+18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

            self.tm.consider_full_frame(
                t,
                frame_bgr,
                (x1i, y1i, x2i, y2i),  # bbox xe ở hệ toạ độ toàn frame (bắt buộc)
                helmet_box,  # có -> tuple 4 số; không có -> None
                lp_box  # có -> tuple 4 số; không có -> None
            )
        # --- Finalize các track bị mất ---
        for tid in lost_ids:
            summary = self.tm.finalize_track(tid)
            if not summary: continue
            if summary["decision"] == "violation":
                incident = self.inc_builder.build_from_summary(summary)
                # Gửi email cảnh báo sau khi tạo incident
                self.send_violation_email(incident)
            elif summary["decision"] == "observation":
                self.obs_builder.build_from_summary(summary)

        # FPS & buffer
        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        self.last_annotated_bgr = annotated.copy()
        return annotated

    # ---------- Display ----------
    def _show_on_canvas(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        c_w = self.canvas.winfo_width() or 800
        c_h = self.canvas.winfo_height() or 600
        img.thumbnail((c_w, c_h), Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(image=img)
        
        # Chỉ update nếu có thay đổi đáng kể hoặc lần đầu
        with self.display_lock:
            if not hasattr(self, '_canvas_img_id') or self._canvas_img_id is None:
                self.canvas.delete("all")
                self._canvas_img_id = self.canvas.create_image(c_w // 2, c_h // 2, image=tkimg, anchor="center")
            else:
                # Cập nhật image và toạ độ (nếu canvas thay đổi kích thước)
                self.canvas.coords(self._canvas_img_id, c_w // 2, c_h // 2)
                self.canvas.itemconfig(self._canvas_img_id, image=tkimg)
            self._last_tkimg = tkimg
            self.tkimg_ref = tkimg

    # ---------- Loops ----------
    def _loop_video_capture(self):
        self.log("Đang chạy video/webcam... Nhấn Stop để dừng.")
        target_fps = 30.0  # FPS mục tiêu
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()
        while self.running and self.cap and self.cap.isOpened():
            current_time = time.time()
            # Đảm bảo không xử lý quá nhanh
            if current_time - last_frame_time < frame_time:
                time.sleep(0.001)  # Sleep ngắn để không block
                continue
            
            ok, frame = self.cap.read()
            if not ok:
                # NEW: video kết thúc hoặc webcam mất khung -> chốt tất cả track
                self.flush_all_tracks()
                break

            annotated = self._predict_draw(frame)
            self._show_on_canvas(annotated)
            self.frame_idx += 1
            last_frame_time = current_time
            
            # Thêm delay nhỏ để tránh nhấp nháy
            time.sleep(0.01)
        self.stop()

    def _run_folder_images(self, folder: Path):
        self.log(f"Đang chạy thư mục: {folder}")
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        paths = [p for p in folder.rglob("*") if p.suffix.lower() in image_exts]
        if not paths:
            messagebox.showinfo("Trống", "Không tìm thấy ảnh trong thư mục.")
            self.stop(); return
        target_fps = 10.0  # FPS thấp hơn cho ảnh
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()
        for p in sorted(paths):
            if not self.running: break
            current_time = time.time()
            if current_time - last_frame_time < frame_time:
                time.sleep(0.01)
                continue
                
            img = cv2.imread(str(p))
            if img is None: continue
            annotated = self._predict_draw(img)
            self._show_on_canvas(annotated)
            self.frame_idx += 1
            last_frame_time = current_time
            time.sleep(0.05)  # Delay giữa các ảnh
        self.stop()

    def _run_single_image(self, path: str):
        self.log(f"Đang chạy ảnh: {path}")
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Lỗi ảnh", "Không đọc được ảnh."); self.stop(); return
        annotated = self._predict_draw(img)
        self._show_on_canvas(annotated)
        self.log("Xong ảnh. (Bạn có thể Lưu khung hiện tại)")
        self.frame_idx += 1
        # NEW: chốt ngay sau khi xử lý ảnh đơn
        self.flush_all_tracks()

    # ---------- Save ----------
    def save_current_frame(self):
        if self.tkimg_ref is None:
            messagebox.showinfo("Chưa có khung hình", "Chưa có hình để lưu.")
            return
        try:
            if hasattr(self, "last_annotated_bgr") and self.last_annotated_bgr is not None:
                save_path = filedialog.asksaveasfilename(
                    title="Lưu ảnh kết quả",
                    defaultextension=".jpg",
                    filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
                )
                if save_path:
                    cv2.imwrite(save_path, self.last_annotated_bgr)
                    messagebox.showinfo("Đã lưu", f"Đã lưu: {save_path}")
                return
        except Exception:
            pass
        messagebox.showinfo("Gợi ý",
                            "Nên lưu self.last_annotated_bgr để có chất lượng gốc.\n"
                            "Hiện đang lưu từ canvas nên chất lượng có thể giảm.")

    # ---------- Buffer property ----------
    @property
    def last_annotated_bgr(self):
        return getattr(self, "_last_annotated_bgr", None)

    @last_annotated_bgr.setter
    def last_annotated_bgr(self, val):
        self._last_annotated_bgr = val

# ============ Main ============
def main():
    root = tk.Tk()
    try:
        if os.name == "nt":
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        style = ttk.Style(); style.theme_use("clam")
    except Exception:
        pass
    app = YoloTesterGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
