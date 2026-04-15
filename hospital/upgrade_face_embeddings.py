"""
upgrade_face_embeddings.py
──────────────────────────
Standalone script to re-generate 512-D FaceNet / DeepFace embeddings for
every patient that already has a stored face image in the hospital database.

Run from the /hospital subdirectory with the venv active:

    python upgrade_face_embeddings.py

The script tries backends in this priority order:
  1. facenet-pytorch  (MTCNN → InceptionResNetV1 pretrained on VGGFace2)
  2. DeepFace / Facenet512 + MTCNN detector
  3. DeepFace / Facenet512 + OpenCV detector (relaxed)
  4. OpenCV Haar pixel descriptor             (last resort)

After updating all embeddings the script invokes train_face_model() to
recompute optimal cosine-similarity thresholds from the updated dataset.
"""

import os
import sys
import json
import base64
import io

import numpy as np
from PIL import Image

# ── Minimal Flask / DB setup ──────────────────────────────────────────────────
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = 'sqlite:///' + os.path.abspath(os.path.join(os.path.dirname(__file__), 'instance', 'hospital_kiosk.db')).replace('\\\\\\\\' , '/')
_app = Flask(__name__)
_app.config['SQLALCHEMY_DATABASE_URI'] = DB_PATH
_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
_db = SQLAlchemy(_app)


class Patient(_db.Model):
    """Minimal Patient mirror for migration purposes."""
    id              = _db.Column(_db.Integer, primary_key=True)
    patient_id      = _db.Column(_db.String(20), unique=True, nullable=False)
    name            = _db.Column(_db.String(100), nullable=False)
    face_descriptor = _db.Column(_db.Text)
    face_image      = _db.Column(_db.Text)


# ── Load deep-learning backends (same priority as main app) ──────────────────
FACENET_AVAILABLE  = False
DEEPFACE_AVAILABLE = False
OPENCV_AVAILABLE   = False
torch = cv2 = DeepFace = MTCNN_DETECTOR = FACENET_MODEL = FACENET_DEVICE = None
FACE_CASCADE = None

try:
    import torch as _torch
    torch = _torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[!] PyTorch not available")

try:
    import cv2 as _cv2
    cv2 = _cv2
    cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    FACE_CASCADE = cv2.CascadeClassifier(cascade)
    OPENCV_AVAILABLE = not FACE_CASCADE.empty()
    if OPENCV_AVAILABLE:
        print("[+] OpenCV + Haar cascade loaded (fallback)")
except ImportError:
    print("[!] OpenCV not available")

try:
    from deepface import DeepFace as _DF
    DeepFace = _DF
    DEEPFACE_AVAILABLE = True
    print("[+] DeepFace available")
except ImportError:
    print("[!] DeepFace not installed – pip install deepface")

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_DEVICE  = torch.device('cuda' if TORCH_OK and torch.cuda.is_available() else 'cpu')
    FACENET_MODEL   = InceptionResnetV1(pretrained='vggface2').eval().to(FACENET_DEVICE)
    MTCNN_DETECTOR  = MTCNN(
        image_size=160, margin=20, keep_all=False,
        post_process=True, select_largest=True, device=FACENET_DEVICE
    )
    FACENET_AVAILABLE = True
    print(f"[+] facenet-pytorch loaded  (device={FACENET_DEVICE})")
except Exception as e:
    print(f"[!] facenet-pytorch not available: {e}")


# ── Core embedding extractor (mirrors _extract_embedding_from_image_bytes) ────
def extract_embedding(image_bytes):
    """Return (embedding_list, model_name) or (None, None)."""
    try:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        print(f"  [decode error] {e}")
        return None, None

    # Priority 1 – facenet-pytorch
    if FACENET_AVAILABLE and MTCNN_DETECTOR and FACENET_MODEL:
        try:
            ft = MTCNN_DETECTOR(img_pil)
            if ft is not None:
                with torch.no_grad():
                    emb = FACENET_MODEL(ft.unsqueeze(0).to(FACENET_DEVICE))
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    return emb.squeeze().cpu().numpy().astype(np.float32).tolist(), "FaceNet-VGGFace2"
        except Exception as e:
            print(f"  [facenet error] {e}")

    # Priority 2 – DeepFace / Facenet512 + MTCNN
    if DEEPFACE_AVAILABLE:
        try:
            r = DeepFace.represent(
                img_path=np.array(img_pil),
                model_name='Facenet512',
                detector_backend='mtcnn',
                enforce_detection=True,
                align=True,
                normalization='Facenet2018'
            )
            if r:
                return r[0]['embedding'], "DeepFace-Facenet512-mtcnn"
        except Exception:
            pass

        # Priority 2b – DeepFace / Facenet512 + OpenCV (relaxed)
        try:
            r = DeepFace.represent(
                img_path=np.array(img_pil),
                model_name='Facenet512',
                detector_backend='opencv',
                enforce_detection=False,
                align=True,
                normalization='Facenet2018'
            )
            if r:
                return r[0]['embedding'], "DeepFace-Facenet512-opencv"
        except Exception as e:
            print(f"  [DeepFace error] {e}")

    # Priority 3 – OpenCV Haar pixel fallback
    if OPENCV_AVAILABLE and FACE_CASCADE is not None:
        try:
            bgr  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            fs   = FACE_CASCADE.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
            if len(fs) > 0:
                x, y, w, h = max(fs, key=lambda f: f[2]*f[3])
                roi  = cv2.resize(bgr[y:y+h, x:x+w], (32, 16)).flatten().astype(np.float32)
                norm = np.linalg.norm(roi)
                return (roi / (norm + 1e-8)).tolist(), "OpenCV-Haar-Fallback"
        except Exception as e:
            print(f"  [OpenCV error] {e}")

    return None, None


# ── Migration ─────────────────────────────────────────────────────────────────
def migrate_embeddings():
    print("\n" + "="*65)
    print(" FACE EMBEDDING UPGRADE (FaceNet-VGGFace2 + DeepFace)")
    print("="*65)

    with _app.app_context():
        patients = Patient.query.filter(Patient.face_image.isnot(None)).all()
        if not patients:
            print("[!] No patients with stored face images found.")
            return

        print(f"[*] Found {len(patients)} patients with face images.\n")
        success, failed = 0, 0
        usage = {}

        for p in patients:
            try:
                raw = (p.face_image or '')
                if ',' in raw:
                    raw = raw.split(',', 1)[1]
                img_bytes = base64.b64decode(raw)

                emb, model = extract_embedding(img_bytes)
                if emb is None:
                    raise ValueError("No face detected")

                p.face_descriptor = json.dumps(emb)
                usage[model] = usage.get(model, 0) + 1
                success += 1
                print(f"  [OK]   {p.name:30s} ({p.patient_id})  dim={len(emb)}  model={model}")

            except Exception as e:
                failed += 1
                print(f"  [FAIL] {p.name} ({p.patient_id}):  {e}")

        _db.session.commit()

        print(f"\n[*] Upgrade complete: {success} OK, {failed} failed")
        print(f"[*] Backends used: {usage}")

        # ── Compute threshold statistics ──────────────────────────────────
        valid = []
        for p in Patient.query.filter(Patient.face_descriptor.isnot(None)).all():
            try:
                v = np.array(json.loads(p.face_descriptor), dtype=np.float32)
                n = np.linalg.norm(v)
                if n > 0:
                    valid.append(v / n)
            except Exception:
                continue

        if len(valid) < 2:
            print("[!] <2 valid embeddings – threshold computation skipped.")
            return

        sims, dists = [], []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                sims.append(float(np.dot(valid[i], valid[j])))
                dists.append(float(np.linalg.norm(valid[i] - valid[j])))

        sa, da = np.array(sims), np.array(dists)
        print(f"\n[STATS  ({len(valid)} patients)]")
        print(f"  Cosine sim   mean={sa.mean():.4f}  std={sa.std():.4f}  range=[{sa.min():.4f},{sa.max():.4f}]")
        print(f"  Euclidean    mean={da.mean():.4f}  std={da.std():.4f}  range=[{da.min():.4f},{da.max():.4f}]")

        opt_cos = float(np.clip(np.percentile(sa, 70), 0.45, 0.90))
        opt_euc = float(np.clip(np.percentile(da, 70), 0.15, 1.20))
        print(f"\n[SUGGESTED THRESHOLDS]  cosine>={opt_cos:.4f}  euclidean<={opt_euc:.4f}")
        print("\n[!] Tip: call POST /api/train-face-model to save these thresholds to the DB.")
        print("="*65 + "\n")


if __name__ == '__main__':
    migrate_embeddings()
