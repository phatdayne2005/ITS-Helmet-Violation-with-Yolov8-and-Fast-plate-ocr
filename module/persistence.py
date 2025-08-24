# persistence.py
import os, json, shutil
from datetime import datetime
from typing import Optional, Dict

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True); return p

def atomic_write(path: str, data: dict):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

class DataPaths:
    def __init__(self, root="data"):
        self.root = ensure_dir(root)
        # vehicles.json là file; không có vehicles_dir. Chỉ cần đảm bảo các thư mục dưới tồn tại.
        ensure_dir(self.tracks_dir)
        ensure_dir(self.incidents_dir)
        ensure_dir(self.observations_dir)

    @property
    def vehicles_path(self): return os.path.join(self.root, "vehicles.json")
    @property
    def tracks_dir(self): return os.path.join(self.root, "tracks")
    @property
    def incidents_dir(self): return os.path.join(self.root, "incidents")
    @property
    def observations_dir(self): return os.path.join(self.root, "observations")
    @property
    def incidents_index(self): return os.path.join(self.root, "incidents", "index.json")

    def new_incident_id(self):
        return "INC_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def new_observation_id(self):
        return "OBS_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

class JsonStore:
    def __init__(self, paths: DataPaths):
        self.paths = paths
        # tạo vehicles.json & incidents/index.json nếu chưa có
        if not os.path.exists(self.paths.vehicles_path):
            atomic_write(self.paths.vehicles_path, {"version":1, "vehicles":[]})
        if not os.path.exists(self.paths.incidents_index):
            atomic_write(self.paths.incidents_index, [])

    def find_owner(self, plate: str) -> Dict:
        db = read_json(self.paths.vehicles_path, {"vehicles":[]})
        norm = plate.upper().replace(" ", "")
        for v in db.get("vehicles", []):
            if v.get("plate", "").upper().replace(" ", "") == norm:
                return {
                    "found": True,
                    "plate": v.get("plate"),
                    "name": v.get("owner", {}).get("name", ""),
                    "email": v.get("owner", {}).get("email", ""),
                    "phone": v.get("owner", {}).get("phone", ""),
                    "vehicle": v.get("vehicle", {})
                }
        return {"found": False, "plate": plate}

class IncidentBuilder:
    def __init__(self, paths: DataPaths, store: JsonStore, camera_id="CAM_01", location="Gate A"):
        self.paths, self.store = paths, store
        self.camera_id, self.location = camera_id, location

    def _copy_if(self, src: Optional[str], dst: str):
        if src and os.path.exists(src):
            ensure_dir(os.path.dirname(dst))
            shutil.copy2(src, dst)
            return True
        return False

    def build_from_summary(self, summary: dict):
        inc_id = self.paths.new_incident_id()
        out_dir = ensure_dir(os.path.join(self.paths.incidents_dir, inc_id))
        ev_dir = ensure_dir(os.path.join(out_dir, "evidence"))
        mb = self._copy_if(summary["best"].get("motorbike"), os.path.join(ev_dir, "motorbike.jpg"))
        he = self._copy_if(summary["best"].get("helmet"), os.path.join(ev_dir, "helmet.jpg"))
        lp = self._copy_if(summary["best"].get("lp_img"), os.path.join(ev_dir, "plate.jpg"))
        # NEW:
        ff = self._copy_if(summary["best"].get("full_frame"), os.path.join(ev_dir, "full_frame.jpg"))
        owner = self.store.find_owner(summary.get("plate_final", ""))

        incident = {
            "id": inc_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "camera": {"id": self.camera_id, "location": self.location},
            "helmet_state": summary["helmet_state"],  # NEW ✅ thêm dòng này
            "track": {
                "track_id": summary["track_id"],
                "frames_observed": summary["frames_observed"],
                "helmet_ema": summary["helmet_ema"],
                "helmet_state": summary["helmet_state"]
            },
            "plate": {
                "text": summary.get("plate_final", ""),
                "confidence": summary.get("plate_conf", 0.0),
                "locked": True
            },
            "evidence": {
                "motorbike": "evidence/motorbike.jpg" if mb else None,
                "helmet": "evidence/helmet.jpg" if he else None,
                "plate": "evidence/plate.jpg" if lp else None,
                "full_frame": "evidence/full_frame.jpg" if ff else None   # NEW
            },
            "email": {
                "status": "queued" if owner.get("found") and owner.get("email") else "skipped"
            },
            "notes": ""
        }
        atomic_write(os.path.join(out_dir, "incident.json"), incident)

        index = read_json(self.paths.incidents_index, [])
        summary_row = {
            "id": inc_id,
            "time": incident["created_at"],
            "plate": incident["plate"]["text"],
            "helmet_state": incident["track"]["helmet_state"],
            "owner_name": owner.get("name", ""),
            "owner_email": owner.get("email", ""),
            "email_status": incident["email"]["status"],
            "email_sent_at": incident["email"].get("sent_at", ""),
            "email_error": incident["email"].get("error", ""),
            "confidence": {"plate": incident["plate"]["confidence"], "helmet_ema": incident["track"]["helmet_ema"]},
            "paths": {"folder": out_dir,
                      "motorbike": os.path.join(out_dir, "evidence/motorbike.jpg") if mb else None,
                      "helmet": os.path.join(out_dir, "evidence/helmet.jpg") if he else None,
                      "plate": os.path.join(out_dir, "evidence/plate.jpg") if lp else None,
                      "full_frame": os.path.join(out_dir, "evidence/full_frame.jpg") if ff else None}
        }
        index.insert(0, summary_row)
        index = index[:500]
        atomic_write(self.paths.incidents_index, index)
        return incident

class ObservationBuilder:
    def __init__(self, paths: DataPaths):
        self.paths = paths

    def _copy_if(self, src: Optional[str], dst: str):
        if src and os.path.exists(src):
            ensure_dir(os.path.dirname(dst))
            shutil.copy2(src, dst)
            return True
        return False

    def build_from_summary(self, summary: dict):
        obs_id = self.paths.new_observation_id()
        out_dir = ensure_dir(os.path.join(self.paths.observations_dir, obs_id))
        ev_dir  = ensure_dir(os.path.join(out_dir, "evidence"))
        def cp(src, name):
            if src and os.path.exists(src):
                ensure_dir(os.path.dirname(os.path.join(ev_dir, name)))
                shutil.copy2(src, os.path.join(ev_dir, name))
                return True
            return False
        mb = cp(summary["best"].get("motorbike"), "motorbike.jpg")
        he = cp(summary["best"].get("helmet"),    "helmet.jpg")
        lp = cp(summary["best"].get("lp_img"),    "plate.jpg")
        ff = cp(summary["best"].get("full_frame"), "full_frame.jpg")

        observation = {
            "id": obs_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "camera": {"id": "CAM_01", "location": "Gate A"},
            "track": {
                "track_id": summary["track_id"],
                "frames_observed": summary["frames_observed"],
                "helmet_ema": summary["helmet_ema"],
                "helmet_state": summary["helmet_state"],
            },
            "plate": {
                "text": summary.get("plate_final",""),
                "confidence": summary.get("plate_conf", 0.0),
                "locked": True
            },
            "evidence": {
                "motorbike": "evidence/motorbike.jpg" if mb else None,
                "helmet":    "evidence/helmet.jpg" if he else None,
                "plate":     "evidence/plate.jpg" if lp else None,
                "full_frame": "evidence/full_frame.jpg" if ff else None  # NEW
            }
        }
        atomic_write(os.path.join(out_dir, "observation.json"), observation)
        return observation
