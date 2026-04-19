from flask import Flask, request, jsonify, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from sqlalchemy import inspect, text
import datetime
import html
import uuid
import json
import os
import base64
import io
import math
import sqlite3
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from PIL import Image
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from sqlalchemy.pool import StaticPool

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# Face AI stack – availability flags (all start False)
DEEPFACE_AVAILABLE = False
FACENET_AVAILABLE = False
OPENCV_AVAILABLE = False
TORCH_AVAILABLE = False

# Ensure optional globals always exist so the rest of the file never NameErrors
cv2 = None
FACE_CASCADE = None
EYE_CASCADE = None
torch = None
DeepFace = None
MTCNN = None
MTCNN_DETECTOR = None
FACENET_MODEL = None
FACENET_DEVICE = None

# ─── Matching constants (NIST-recommended for FaceNet-VGGFace2 512-D) ──────────
FACE_EMBEDDING_DIM       = 512
FACENET_IMAGE_SIZE       = 160
# Cosine-similarity thresholds (higher = stricter)
FACE_MATCH_EXCELLENT_THRESHOLD = 0.80   # near-perfect match
DEFAULT_COSINE_THRESHOLD       = 0.68   # standard match gate
DEFAULT_SIMILARITY_GAP         = 0.04   # min gap vs 2nd-best
# Euclidean distance thresholds (L2-norm of unit vectors, max √2 ≈ 1.414)
DEFAULT_EUCLIDEAN_THRESHOLD    = 0.85   # tighter for FaceNet 512-D unit vecs
IMAGE_RESAMPLING = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def cosine_to_euclidean_threshold(cosine_threshold):
    """Convert cosine-similarity threshold to L2-distance threshold for unit vectors."""
    return float(math.sqrt(max(0.0, 2.0 - (2.0 * cosine_threshold))))


try:
    import torch
    TORCH_AVAILABLE = True
    print("[+] PyTorch available")
except ImportError:
    print("[!] PyTorch not available")

try:
    import cv2
    OPENCV_AVAILABLE = True
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    EYE_CASCADE = cv2.CascadeClassifier(eye_cascade_path)

    if not FACE_CASCADE.empty():
        print("[+] OpenCV available with Face and Eye Cascades for fallback detection")
    else:
        OPENCV_AVAILABLE = False
        print("[!] OpenCV Haar Cascade failed to load")
except ImportError:
    print("[!] OpenCV not available")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("[+] DeepFace available (face detection + recognition backend)")
except ImportError:
    print("[!] DeepFace not installed – pip install deepface")

try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
    FACENET_AVAILABLE = True
    print("[+] facenet-pytorch available")
    try:
        FACENET_DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        FACENET_MODEL = InceptionResnetV1(pretrained='vggface2').eval().to(FACENET_DEVICE)
        MTCNN_DETECTOR = MTCNN(
            image_size=FACENET_IMAGE_SIZE,
            margin=20,          # slightly wider crop for better alignment
            keep_all=False,     # only the largest / most confident face
            post_process=True,  # return normalised tensor ready for FaceNet
            select_largest=True,
            device=FACENET_DEVICE
        )
        if TORCH_AVAILABLE and torch.cuda.is_available():
            print("[+] FaceNet running on GPU")
        else:
            print("[+] FaceNet running on CPU")
        print("[+] FaceNet InceptionResNetV1 (VGGFace2) + MTCNN detector loaded")
    except Exception as _e:
        print(f"[!] FaceNet model load failed: {_e}")
        FACENET_AVAILABLE = False
        FACENET_MODEL = None
        MTCNN_DETECTOR = None
        FACENET_DEVICE = None
except ImportError:
    print("[!] facenet-pytorch not installed – pip install facenet-pytorch")


# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hospital-kiosk-secret-key-2024'


def _normalize_database_url(raw_url):
    """Normalize database URLs that Flask-SQLAlchemy can consume."""
    raw_url = (raw_url or '').strip()
    if not raw_url:
        return None
    if raw_url.startswith('postgres://'):
        return 'postgresql+psycopg2://' + raw_url[len('postgres://'):]
    if raw_url.startswith('postgresql://') and '+psycopg2' not in raw_url:
        return 'postgresql+psycopg2://' + raw_url[len('postgresql://'):]
    return raw_url


def _resolve_database_uri():
    """Pick the best database backend for the current environment."""
    def _can_write_sqlite(db_file):
        probe = sqlite3.connect(db_file, timeout=2)
        try:
            probe.execute('BEGIN IMMEDIATE')
            probe.execute('CREATE TABLE IF NOT EXISTS __codex_write_probe__(id INTEGER)')
            probe.execute('INSERT INTO __codex_write_probe__ DEFAULT VALUES')
            probe.rollback()
            return True
        finally:
            probe.close()

    configured_url = _normalize_database_url(os.environ.get('DATABASE_URL'))
    if configured_url:
        print("[OK] Using DATABASE_URL from environment")
        return configured_url, None, False

    configured_path = os.environ.get('HOSPITAL_KIOSK_DB_PATH')
    if configured_path:
        db_file = os.path.abspath(configured_path)
        os.makedirs(os.path.dirname(db_file) or '.', exist_ok=True)
        try:
            if _can_write_sqlite(db_file):
                return f"sqlite:///{db_file.replace('\\', '/')}", db_file, False
            print(f"[!] SQLite file database unavailable at {db_file}: write probe failed")
        except Exception as exc:
            print(f"[!] SQLite write probe failed for {db_file}: {exc}")

    legacy_db_path = os.path.abspath(os.path.join(app.root_path, 'instance', 'hospital_kiosk.db'))
    try:
        os.makedirs(os.path.dirname(legacy_db_path) or '.', exist_ok=True)
        if _can_write_sqlite(legacy_db_path):
            return f"sqlite:///{legacy_db_path.replace('\\', '/')}", legacy_db_path, False
        print(f"[!] SQLite file database unavailable at {legacy_db_path}: write probe failed")
    except Exception as exc:
        print(f"[!] SQLite file database unavailable at {legacy_db_path}: {exc}")

    print("[!] Falling back to in-memory SQLite for this session")
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'connect_args': {'check_same_thread': False},
        'poolclass': StaticPool,
    }
    return 'sqlite://', None, True


db_uri, db_path, USING_IN_MEMORY_DB = _resolve_database_uri()
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

if USING_IN_MEMORY_DB:
    print("[!] In-memory SQLite active; data will not survive restart")
elif db_uri.startswith('sqlite:///'):
    print(f"[OK] SQLite database file: {db_path}")
else:
    print("[OK] External database backend configured")

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'doctor_login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='patient')
    full_name = db.Column(db.String(100))
    department = db.Column(db.String(50))
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100))
    aadhaar_number = db.Column(db.String(20))  # Optional Aadhaar number
    department = db.Column(db.String(50), nullable=False)
    consultation_type = db.Column(db.String(20), nullable=False)
    consultation_date = db.Column(db.Date)
    consultation_time = db.Column(db.Time)
    queue_number = db.Column(db.String(20))
    check_in_time = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    status = db.Column(db.String(20), default='waiting')
    doctor_assigned = db.Column(db.String(100))
    completion_time = db.Column(db.DateTime)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    face_descriptor = db.Column(db.Text)  # JSON string of face embedding
    face_image = db.Column(db.Text)  # Base64 encoded face image

class CapturedFace(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    face_descriptor = db.Column(db.Text, nullable=False)
    face_image = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Department(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dept_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100))
    floor = db.Column(db.Integer)
    description = db.Column(db.Text)
    icon = db.Column(db.String(50))

class EmergencyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    emergency_id = db.Column(db.String(20), unique=True, nullable=False)
    patient_id = db.Column(db.String(20), db.ForeignKey('patient.patient_id'))
    patient_name = db.Column(db.String(100))
    department = db.Column(db.String(50))
    declared_by = db.Column(db.String(100))
    severity = db.Column(db.String(20))
    reason = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    resolved = db.Column(db.Boolean, default=False)
    resolved_by = db.Column(db.String(100))
    resolved_at = db.Column(db.DateTime)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    sender = db.Column(db.String(20), nullable=False)  # 'user' or 'bot'
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    intent = db.Column(db.String(50))  # Store detected intent

class FaceRecognitionModel(db.Model):
    """Store face recognition model parameters and metrics"""
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), default='RobustMultiScale')
    cosine_threshold = db.Column(db.Float, default=0.65)
    euclidean_threshold = db.Column(db.Float, default=10.0)
    similarity_gap = db.Column(db.Float, default=0.05)
    last_trained = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    total_patients = db.Column(db.Integer, default=0)
    accuracy = db.Column(db.Float, default=0.0)
    precision = db.Column(db.Float, default=0.0)
    recall = db.Column(db.Float, default=0.0)
    false_positives = db.Column(db.Integer, default=0)
    false_negatives = db.Column(db.Integer, default=0)
    training_data = db.Column(db.Text)  # JSON with training statistics


def parse_consultation_date(value):
    """Parse the consultation date submitted from the registration form."""
    return datetime.datetime.strptime(value, '%Y-%m-%d').date()


def format_consultation_date(value):
    """Return a human-friendly consultation date."""
    if not value:
        return 'Not scheduled'
    return value.strftime('%d %b %Y')


def format_consultation_time(value):
    """Return a human-friendly consultation time."""
    if not value:
        return 'Not scheduled'
    return value.strftime('%I:%M %p')


def get_consultation_slot_minutes():
    """Average consultation duration used for queue-based time prediction."""
    try:
        return max(5, int(os.environ.get('CONSULTATION_SLOT_MINUTES', '15')))
    except ValueError:
        return 15


def get_opd_start_time():
    """Start time used when scheduling future consultations."""
    raw = os.environ.get('OPD_START_TIME', '09:00').strip()
    try:
        return datetime.datetime.strptime(raw, '%H:%M').time()
    except ValueError:
        return datetime.time(9, 0)


def estimate_consultation_slot(department, consultation_date):
    """Predict consultation time from the active queue for a department and day."""
    active_statuses = ['scheduled', 'waiting', 'emergency', 'in_progress']
    patients_ahead = Patient.query.filter(
        Patient.department == department,
        Patient.consultation_date == consultation_date,
        Patient.status.in_(active_statuses)
    ).count()

    slot_minutes = get_consultation_slot_minutes()
    base_dt = datetime.datetime.combine(consultation_date, get_opd_start_time())

    if consultation_date == datetime.date.today():
        immediate_start = datetime.datetime.now() + datetime.timedelta(minutes=slot_minutes)
        if immediate_start > base_dt:
            base_dt = immediate_start

    estimated_dt = base_dt + datetime.timedelta(minutes=patients_ahead * slot_minutes)
    return estimated_dt.time(), patients_ahead + 1


def format_patient_service_time(patient):
    """Show the queue-predicted consultation slot for active visits."""
    if patient.status == 'scheduled':
        schedule_parts = []
        if patient.consultation_date:
            schedule_parts.append(format_consultation_date(patient.consultation_date))
        if patient.consultation_time:
            schedule_parts.append(format_consultation_time(patient.consultation_time))
        if schedule_parts:
            return f"Scheduled: {', '.join(schedule_parts)}"

    if patient.consultation_time and patient.status in ['waiting', 'emergency', 'in_progress']:
        return f"Est. {format_consultation_time(patient.consultation_time)}"

    if patient.check_in_time:
        return patient.check_in_time.strftime('%I:%M %p')
    schedule_parts = []
    if patient.consultation_date:
        schedule_parts.append(format_consultation_date(patient.consultation_date))
    if patient.consultation_time:
        schedule_parts.append(format_consultation_time(patient.consultation_time))
    if schedule_parts:
        return f"Scheduled: {', '.join(schedule_parts)}"
    return 'Pending'


def format_schedule_slot(patient):
    """Return a concise date/time summary for confirmations and SMS."""
    schedule_parts = []
    if patient.consultation_date:
        schedule_parts.append(format_consultation_date(patient.consultation_date))
    if patient.consultation_time:
        schedule_parts.append(format_consultation_time(patient.consultation_time))
    return ', '.join(schedule_parts) if schedule_parts else 'Not scheduled'


def ensure_patient_schema():
    """Apply small schema upgrades without a migration framework."""
    patient_columns = {
        column['name'] for column in inspect(db.engine).get_columns('patient')
    }
    if 'consultation_date' not in patient_columns:
        db.session.execute(text("ALTER TABLE patient ADD COLUMN consultation_date DATE"))
        db.session.commit()
        print("[OK] Added patient.consultation_date column")
    if 'consultation_time' not in patient_columns:
        db.session.execute(text("ALTER TABLE patient ADD COLUMN consultation_time TIME"))
        db.session.commit()
        print("[OK] Added patient.consultation_time column")

    updated_rows = False
    patients = Patient.query.filter(
        (Patient.consultation_date.is_(None)) | (Patient.consultation_time.is_(None))
    ).all()
    for patient in patients:
        if not patient.check_in_time:
            continue

        if patient.consultation_date is None:
            patient.consultation_date = patient.check_in_time.date()
            updated_rows = True

        if patient.consultation_time is None:
            patient.consultation_time = patient.check_in_time.time().replace(microsecond=0)
            updated_rows = True

    if updated_rows:
        db.session.commit()


def normalize_mobile_number(phone):
    """Convert local mobile numbers into a likely E.164 format for SMS APIs."""
    raw = (phone or '').strip()
    digits = ''.join(ch for ch in raw if ch.isdigit())
    default_cc = os.environ.get('SMS_DEFAULT_COUNTRY_CODE', '+91').strip() or '+91'
    if not default_cc.startswith('+'):
        default_cc = f'+{default_cc}'

    if raw.startswith('+') and digits:
        return f'+{digits}'
    if len(digits) == 10:
        return f'{default_cc}{digits}'
    if len(digits) > 10:
        return f'+{digits}'
    return raw


def build_registration_sms(patient):
    """Build the registration confirmation SMS body."""
    queue_text = patient.queue_number or 'Will be assigned on arrival'
    return (
        "Smart Hospital registration confirmed.\n"
        f"Patient ID: {patient.patient_id}\n"
        f"Name: {patient.name}\n"
        f"Department: {patient.department}\n"
        f"Visit Type: {patient.consultation_type.replace('-', ' ').title()}\n"
        f"Scheduled Date: {format_consultation_date(patient.consultation_date)}\n"
        f"Estimated Consultation Time: {format_consultation_time(patient.consultation_time)}\n"
        f"Status: {patient.status.replace('_', ' ').title()}\n"
        f"Queue: {queue_text}"
    )


def sms_is_configured():
    """Check whether the configured SMS provider has the required credentials."""
    provider = os.environ.get('SMS_PROVIDER', 'twilio').strip().lower()
    if provider == 'twilio':
        required = [
            os.environ.get('TWILIO_ACCOUNT_SID', '').strip(),
            os.environ.get('TWILIO_AUTH_TOKEN', '').strip(),
            os.environ.get('TWILIO_FROM_NUMBER', '').strip(),
        ]
        return all(required)

    if provider == 'textbee':
        required = [
            os.environ.get('TEXTBEE_API_KEY', '').strip(),
            os.environ.get('TEXTBEE_DEVICE_ID', '').strip(),
        ]
        return all(required)

    return False


def send_registration_sms(patient):
    """Send a registration confirmation SMS when provider credentials are available."""
    provider = os.environ.get('SMS_PROVIDER', 'twilio').strip().lower()
    if not sms_is_configured():
        return {'sent': False, 'reason': 'sms_not_configured'}

    to_number = normalize_mobile_number(patient.phone)
    body = build_registration_sms(patient)

    if provider == 'twilio':
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID', '').strip()
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN', '').strip()
        from_number = os.environ.get('TWILIO_FROM_NUMBER', '').strip()

        post_body = urlencode({
            'To': to_number,
            'From': from_number,
            'Body': body,
        }).encode('utf-8')

        auth_header = base64.b64encode(f"{account_sid}:{auth_token}".encode('utf-8')).decode('ascii')
        req = Request(
            url=f'https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json',
            data=post_body,
            headers={
                'Authorization': f'Basic {auth_header}',
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            method='POST'
        )

        with urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode('utf-8'))
        return {
            'sent': True,
            'provider': 'twilio',
            'sid': payload.get('sid'),
            'to': to_number,
        }

    if provider == 'textbee':
        api_key = os.environ.get('TEXTBEE_API_KEY', '').strip()
        device_id = os.environ.get('TEXTBEE_DEVICE_ID', '').strip()
        payload = {
            'recipients': [to_number],
            'message': body,
        }
        req = Request(
            url=f'https://api.textbee.dev/api/v1/gateway/devices/{device_id}/send-sms',
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'x-api-key': api_key,
                'Content-Type': 'application/json',
            },
            method='POST'
        )

        with urlopen(req, timeout=15) as response:
            response_payload = json.loads(response.read().decode('utf-8'))
        return {
            'sent': True,
            'provider': 'textbee',
            'response': response_payload,
            'to': to_number,
        }

    return {'sent': False, 'reason': f'unsupported_provider:{provider}'}

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ──────────────────────────────────────────────────────────────────────────────
# CORE HELPER: extract a FaceNet embedding from raw image bytes
# Priority: facenet-pytorch > DeepFace/Facenet512 > OpenCV pixel fallback
# ──────────────────────────────────────────────────────────────────────────────

def _extract_embedding_from_image_bytes(image_bytes):
    """
    Given raw bytes of an image, return a normalised 512-dim np.float32 list
    embedding and the model name used.
    Returns (embedding_list, model_name) or (None, None) if all backends fail.
    """
    try:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        print(f"[EMBED] Failed to decode image bytes: {e}")
        return None, None

    # ── Priority 1: facenet-pytorch (MTCNN → InceptionResNetV1 VGGFace2) ──
    if FACENET_AVAILABLE and MTCNN_DETECTOR is not None and FACENET_MODEL is not None:
        try:
            face_tensor = MTCNN_DETECTOR(img_pil)   # aligned, normalised tensor, or None
            if face_tensor is not None:
                with torch.no_grad():
                    emb = FACENET_MODEL(face_tensor.unsqueeze(0).to(FACENET_DEVICE))
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    return emb.squeeze().cpu().numpy().astype(np.float32).tolist(), "FaceNet-VGGFace2"
            else:
                print("[EMBED] MTCNN: no face detected, trying DeepFace...")
        except Exception as e:
            print(f"[EMBED] facenet-pytorch error: {e}")

    # ── Priority 2a: DeepFace with MTCNN detector (Facenet512, 512-D) ──────
    if DEEPFACE_AVAILABLE:
        try:
            img_np = np.array(img_pil)
            results = DeepFace.represent(
                img_path=img_np,
                model_name='Facenet512',
                detector_backend='mtcnn',
                enforce_detection=True,
                align=True,
                normalization='Facenet2018'
            )
            if results:
                return results[0]['embedding'], "DeepFace-Facenet512-mtcnn"
        except Exception:
            pass  # fall through to opencv

        # ── Priority 2b: DeepFace with opencv detector (relaxed detection) ─
        try:
            img_np = np.array(img_pil)
            results = DeepFace.represent(
                img_path=img_np,
                model_name='Facenet512',
                detector_backend='opencv',
                enforce_detection=False,
                align=True,
                normalization='Facenet2018'
            )
            if results:
                return results[0]['embedding'], "DeepFace-Facenet512-opencv"
        except Exception as e:
            print(f"[EMBED] DeepFace error: {e}")

    # ── Priority 3: OpenCV Haar + pixel descriptor (512-D, last resort) ─────
    if OPENCV_AVAILABLE and FACE_CASCADE is not None:
        try:
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces  = FACE_CASCADE.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                roi  = cv2.resize(img_bgr[y:y+h, x:x+w], (32, 16)).flatten().astype(np.float32)
                norm = np.linalg.norm(roi)
                roi  = roi / (norm + 1e-8)
                return roi.tolist(), "OpenCV-Haar-Fallback"
        except Exception as e:
            print(f"[EMBED] OpenCV fallback error: {e}")

    return None, None


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN FACE MODEL
# Iterates all patients with stored face images, re-generates high-quality
# FaceNet / DeepFace embeddings, saves them, and computes dataset-level
# cosine-similarity statistics to derive recommended matching thresholds.
# ──────────────────────────────────────────────────────────────────────────────

def train_face_model():
    """
    Re-generate FaceNet/DeepFace embeddings for every patient that has a
    stored face image, update their face_descriptor in the DB, then compute
    cosine-similarity statistics and save recommended thresholds to
    FaceRecognitionModel.
    """
    print("\n" + "="*65)
    print(" FACE RECOGNITION – MODEL TRAINING (FaceNet + DeepFace)")
    print("="*65)

    try:
        with app.app_context():
            patients_with_images = Patient.query.filter(
                Patient.face_image.isnot(None)
            ).all()

            if not patients_with_images:
                print("[!] No patients with stored face images. Register patients first.")
                return None

            print(f"[*] Found {len(patients_with_images)} patients with face images.")
            print("[*] Re-generating embeddings\u2026\n")

            success, failed = 0, 0
            model_usage = {}

            for patient in patients_with_images:
                try:
                    raw = patient.face_image or ''
                    if ',' in raw:
                        raw = raw.split(',', 1)[1]
                    image_bytes = base64.b64decode(raw)

                    embedding, model_name = _extract_embedding_from_image_bytes(image_bytes)

                    if embedding is None:
                        raise ValueError("No face detected in stored image")

                    patient.face_descriptor = json.dumps(embedding)
                    model_usage[model_name] = model_usage.get(model_name, 0) + 1
                    success += 1
                    print(f"  [OK]   {patient.name:30s} ({patient.patient_id})  "
                          f"dim={len(embedding)}  model={model_name}")

                except Exception as e:
                    failed += 1
                    print(f"  [FAIL] {patient.name} ({patient.patient_id}): {e}")

            db.session.commit()
            print(f"\n[*] Done: {success} OK, {failed} failed  |  backends used: {model_usage}")

            # ── Compute pairwise cosine statistics on newly stored embeddings ──
            valid_embs = []
            for p in Patient.query.filter(Patient.face_descriptor.isnot(None)).all():
                try:
                    e = np.array(json.loads(p.face_descriptor), dtype=np.float32)
                    n = np.linalg.norm(e)
                    if n > 0:
                        valid_embs.append(e / n)
                except Exception:
                    continue

            if len(valid_embs) < 2:
                print("[!] <2 valid embeddings; cannot compute thresholds.")
                return None

            sims, dists = [], []
            for i in range(len(valid_embs)):
                for j in range(i + 1, len(valid_embs)):
                    sims.append(float(np.dot(valid_embs[i], valid_embs[j])))
                    dists.append(float(np.linalg.norm(valid_embs[i] - valid_embs[j])))

            sim_arr  = np.array(sims,  dtype=np.float32)
            dist_arr = np.array(dists, dtype=np.float32)

            print(f"\n[STATISTICS ({len(valid_embs)} patients, {len(sims)} pairs)]")
            print(f"  Cosine sim   | mean={sim_arr.mean():.4f}  std={sim_arr.std():.4f}  "
                  f"range=[{sim_arr.min():.4f}, {sim_arr.max():.4f}]")
            print(f"  Euclidean    | mean={dist_arr.mean():.4f}  std={dist_arr.std():.4f}  "
                  f"range=[{dist_arr.min():.4f}, {dist_arr.max():.4f}]")

            # Gate at the 70th percentile of inter-person similarities
            # (same-identity pairs will score well above this)
            opt_cos  = float(np.clip(np.percentile(sim_arr,  70), 0.45, 0.90))
            opt_euc  = float(np.clip(np.percentile(dist_arr, 70), 0.15, 1.20))

            print(f"\n[THRESHOLDS]  cosine>={opt_cos:.4f}  euclidean<={opt_euc:.4f}")

            # ── Persist ──
            model_row = _get_active_face_model_row()
            if not model_row:
                model_row = FaceRecognitionModel()

            model_row.model_name          = 'FaceNet-VGGFace2+DeepFace'
            model_row.cosine_threshold    = opt_cos
            model_row.euclidean_threshold = opt_euc
            model_row.similarity_gap      = DEFAULT_SIMILARITY_GAP
            model_row.last_trained        = datetime.datetime.utcnow()
            model_row.total_patients      = len(valid_embs)
            model_row.accuracy            = float(success) / max(len(patients_with_images), 1)
            model_row.training_data       = json.dumps({
                'method':              'FaceNet-VGGFace2+DeepFace',
                'total_patients':      len(valid_embs),
                'images_processed':    success,
                'images_failed':       failed,
                'models_used':         model_usage,
                'cosine_mean':         float(sim_arr.mean()),
                'cosine_std':          float(sim_arr.std()),
                'euclidean_mean':      float(dist_arr.mean()),
                'euclidean_std':       float(dist_arr.std()),
                'cosine_threshold':    opt_cos,
                'euclidean_threshold': opt_euc,
            })

            db.session.add(model_row)
            db.session.commit()

            result_dict = {
                'cosine_threshold': model_row.cosine_threshold,
                'euclidean_threshold': model_row.euclidean_threshold,
                'total_patients': model_row.total_patients
            }

            print("\n[OK] Face model trained and thresholds saved.")
            print("="*65 + "\n")
            return result_dict

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# Legacy training functions removed.

def _get_active_face_model_row():
    """Prefer a trained model row; otherwise fall back to the oldest default row."""
    trained = FaceRecognitionModel.query.filter(
        FaceRecognitionModel.total_patients > 0
    ).order_by(
        FaceRecognitionModel.last_trained.desc(),
        FaceRecognitionModel.id.desc()
    ).first()
    if trained:
        return trained

    return FaceRecognitionModel.query.order_by(FaceRecognitionModel.id.asc()).first()


def get_face_model_config():
    """Get current face recognition model configuration"""
    with app.app_context():
        model = _get_active_face_model_row()
        if model:
            return {
                'cosine_threshold': model.cosine_threshold,
                'euclidean_threshold': model.euclidean_threshold,
                'similarity_gap': model.similarity_gap
            }
    
    # Return defaults if not found
    return {
        'cosine_threshold': 0.65,
        'euclidean_threshold': 10.0,
        'similarity_gap': 0.05
    }


def _normalize_face_embedding(raw_embedding):
    """Return a unit-length numpy embedding, or None when the input is invalid."""
    try:
        if raw_embedding is None:
            return None
        if isinstance(raw_embedding, str):
            raw_embedding = json.loads(raw_embedding)

        embedding = np.array(raw_embedding, dtype=np.float32)
        if embedding.ndim != 1 or embedding.size == 0:
            return None

        norm = np.linalg.norm(embedding)
        if norm < 1e-6:
            return None

        return embedding / norm
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _match_patient_by_embedding(probe_embedding):
    """Find the best confirmed patient match for a captured face embedding."""
    probe_unit = _normalize_face_embedding(probe_embedding)
    if probe_unit is None:
        return {
            'found': False,
            'error': 'Invalid (zero-norm) embedding received'
        }

    probe_dim = len(probe_unit)

    cfg = get_face_model_config()
    cosine_th = cfg['cosine_threshold']
    euclidean_th = cfg['euclidean_threshold']
    gap_th = cfg['similarity_gap']

    patients = Patient.query.filter(Patient.face_descriptor.isnot(None)).all()
    all_scores = []

    for patient in patients:
        stored_unit = _normalize_face_embedding(patient.face_descriptor)
        if stored_unit is None or len(stored_unit) != probe_dim:
            continue

        cos = float(np.dot(probe_unit, stored_unit))
        dist = float(np.linalg.norm(probe_unit - stored_unit))
        all_scores.append((cos, dist, patient))

    all_scores.sort(key=lambda item: item[0], reverse=True)

    print(f"\n[FACE MATCH] probe_dim={probe_dim}  candidates={len(all_scores)}")
    for cos, dist, patient in all_scores[:5]:
        print(f"  {patient.name:30s}  cosine={cos:.4f}  euclidean={dist:.4f}")

    if not all_scores:
        return {
            'found': False,
            'message': 'No registered patients found in database.',
            'matching_thresholds': {
                'cosine': cosine_th,
                'euclidean': euclidean_th,
                'gap': gap_th,
            },
        }

    best_cos, best_dist, best_patient = all_scores[0]
    second_cos = all_scores[1][0] if len(all_scores) > 1 else -1.0
    gap = best_cos - second_cos

    excellent = best_cos >= FACE_MATCH_EXCELLENT_THRESHOLD
    std_cosine = best_cos >= cosine_th
    std_dist = best_dist <= euclidean_th
    confirmed = excellent or (std_cosine and std_dist)

    print(f"  Thresholds  cosine>={cosine_th}  euclidean<={euclidean_th}  gap>={gap_th}")
    print(f"  excellent={excellent}  cosine={std_cosine}  dist={std_dist}")
    print(f"  RESULT: {'MATCH [OK]' if confirmed else 'NO MATCH [X]'}  ({best_patient.name})")

    return {
        'found': confirmed,
        'best_patient': best_patient,
        'best_similarity': round(best_cos, 4),
        'best_distance': round(best_dist, 4),
        'similarity_gap': round(gap, 4),
        'confidence': {
            'cosine_similarity': round(best_cos, 4),
            'euclidean_distance': round(best_dist, 4),
            'similarity_gap': round(gap, 4),
            'is_excellent_match': excellent,
        },
        'matching_thresholds': {
            'cosine': cosine_th,
            'euclidean': euclidean_th,
            'gap': gap_th,
        }
    }


def _normalize_digit_string(value):
    """Keep only digits so phone/Aadhaar comparisons survive formatting changes."""
    return ''.join(ch for ch in (value or '') if ch.isdigit())


def _normalize_phone_lookup(phone):
    """Return a comparison-friendly phone key, typically the last 10 digits."""
    digits = _normalize_digit_string(phone)
    if len(digits) >= 10:
        return digits[-10:]
    return digits


def _normalize_aadhaar_lookup(aadhaar_number):
    """Return a comparison-friendly Aadhaar key."""
    digits = _normalize_digit_string(aadhaar_number)
    if len(digits) >= 12:
        return digits[-12:]
    return digits


def _find_existing_patient_for_registration(existing_patient_id=None, aadhaar_number=None,
                                            phone=None, face_descriptor=None, name=None):
    """
    Resolve an existing patient record using stable identifiers first.

    Priority:
      1. explicit existing_patient_id from the frontend
      2. Aadhaar number
      3. normalized phone number
      4. face embedding match
    """
    if existing_patient_id:
        patient = Patient.query.filter_by(patient_id=existing_patient_id).first()
        if patient:
            return patient, 'existing_patient_id'

    aadhaar_key = _normalize_aadhaar_lookup(aadhaar_number)
    if aadhaar_key:
        aadhaar_matches = Patient.query.filter(Patient.aadhaar_number.isnot(None)).all()
        for patient in sorted(aadhaar_matches, key=lambda p: p.id, reverse=True):
            if _normalize_aadhaar_lookup(patient.aadhaar_number) == aadhaar_key:
                return patient, 'aadhaar_number'

    phone_key = _normalize_phone_lookup(phone)
    if phone_key:
        phone_matches = Patient.query.filter(Patient.phone.isnot(None)).all()
        same_phone = []
        for patient in phone_matches:
            if _normalize_phone_lookup(patient.phone) == phone_key:
                same_phone.append(patient)

        if same_phone:
            if name:
                name_key = name.strip().lower()
                same_name = [
                    patient for patient in same_phone
                    if (patient.name or '').strip().lower() == name_key
                ]
                if same_name:
                    return sorted(same_name, key=lambda p: p.id, reverse=True)[0], 'phone+name'

            return sorted(same_phone, key=lambda p: p.id, reverse=True)[0], 'phone'

    if face_descriptor:
        match_result = _match_patient_by_embedding(face_descriptor)
        if match_result.get('found') and match_result.get('best_patient') is not None:
            return match_result['best_patient'], 'face'

    return None, None

def init_db():
    """Initialize database with sample data"""
    with app.app_context():
        # Drop all existing tables (COMMENTED OUT FOR DATA PERSISTENCE)
        # print("[*] Resetting database...")
        # try:
        #     db.drop_all()
        # except:
        #     pass
        
        # Create all tables
        db.create_all()
        print("[OK] Database tables created")
        ensure_patient_schema()
        
        # Add sample departments (only if none exist)
        if Department.query.count() == 0:
            departments = [
                Department(
                    dept_id="DEPT001",
                    name="Gynaecology",
                    location="Main Building, First Floor",
                    floor=1,
                    description="Women's health and reproductive care",
                    icon="fas fa-female"
                ),
                Department(
                    dept_id="DEPT002",
                    name="Cardiology",
                    location="Main Building, Second Floor",
                    floor=2,
                    description="Heart and cardiovascular care",
                    icon="fas fa-heartbeat"
                ),
                Department(
                    dept_id="DEPT003",
                    name="Orthopedics",
                    location="Main Building, Second Floor",
                    floor=1,
                    description="Bone, joint, and muscle care",
                    icon="fas fa-bone"
                ),
                Department(
                    dept_id="DEPT004",
                    name="Emergency",
                    location="Main Building, Ground Floor",
                    floor=0,
                    description="24/7 emergency medical care",
                    icon="fas fa-ambulance"
                ),
                Department(
                    dept_id="DEPT005",
                    name="Pediatrics",
                    location="Main Building, First Floor",
                    floor=2,
                    description="Child healthcare",
                    icon="fas fa-baby"
                ),
                Department(
                    dept_id="DEPT006",
                    name="Pharmacy",
                    location="Main Building, Lobby",
                    floor=0,
                    description="Prescription medications",
                    icon="fas fa-pills"
                )
            ]
            
            for dept in departments:
                db.session.add(dept)
            
            user_count_before = User.query.count()

            # Create admin user on a fresh database.
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(username='admin', full_name='Administrator', role='admin')
                admin.set_password('admin123')
                db.session.add(admin)

            # Seed sample doctors only when the user table is empty.
            # This keeps admin deletions persistent across restarts.
            if user_count_before == 0:
                doctors = [
                    # Cardiology Doctors
                    {'username': 'dr_smith', 'full_name': 'Dr. John Smith', 'department': 'Cardiology'},
                    {'username': 'dr_miller', 'full_name': 'Dr. Robert Miller', 'department': 'Cardiology'},
                    {'username': 'dr_chen', 'full_name': 'Dr. Lisa Chen', 'department': 'Cardiology'},

                    # Emergency Doctors
                    {'username': 'dr_jones', 'full_name': 'Dr. Sarah Jones', 'department': 'Emergency'},
                    {'username': 'dr_kumar', 'full_name': 'Dr. Rajesh Kumar', 'department': 'Emergency'},

                    # Gynaecology Doctors
                    {'username': 'dr_wang', 'full_name': 'Dr. Emily Wang', 'department': 'Gynaecology'},
                    {'username': 'dr_sharma', 'full_name': 'Dr. Priya Sharma', 'department': 'Gynaecology'},
                    {'username': 'dr_jackson', 'full_name': 'Dr. Maria Jackson', 'department': 'Gynaecology'},

                    # Orthopedics Doctors
                    {'username': 'dr_patel', 'full_name': 'Dr. Raj Patel', 'department': 'Orthopedics'},
                    {'username': 'dr_rodriguez', 'full_name': 'Dr. Carlos Rodriguez', 'department': 'Orthopedics'},

                    # Pediatrics Doctors
                    {'username': 'dr_anderson', 'full_name': 'Dr. David Anderson', 'department': 'Pediatrics'},
                    {'username': 'dr_gupta', 'full_name': 'Dr. Anjali Gupta', 'department': 'Pediatrics'},

                    # Pharmacy Staff
                    {'username': 'pharm_lee', 'full_name': 'Ms. Jennifer Lee', 'department': 'Pharmacy'},
                    {'username': 'pharm_wilson', 'full_name': 'Mr. James Wilson', 'department': 'Pharmacy'},
                ]
                
                for doctor_data in doctors:
                    doctor = User(
                        username=doctor_data['username'],
                        full_name=doctor_data['full_name'],
                        department=doctor_data['department'],
                        role='doctor' if doctor_data['username'].startswith('dr_') else 'staff'
                    )
                    doctor.set_password('doctor123')
                    db.session.add(doctor)
                print("[OK] Sample data added with multiple doctors per department")
                db.session.commit()
        # Initialize the default face recognition model only once.
        if FaceRecognitionModel.query.count() == 0:
            model = FaceRecognitionModel(
                model_name='RobustMultiScale',
                cosine_threshold=0.65,
                euclidean_threshold=10.0,
                similarity_gap=0.05
            )
            db.session.add(model)
            db.session.commit()
            print("[OK] Face recognition model initialized")
        else:
            print("[OK] Face recognition model already initialized")



def get_base_html(title="Hospital Kiosk", content="", show_nav=True, show_chat=True):
    """Generate complete HTML page with navigation, chatbot, and responsive design"""
    # Dynamic Login/Dashboard Link based on User Role
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            login_link = f'<li class="nav-item"><a class="nav-link fw-bold text-primary px-3" href="/admin/dashboard"><i class="fas fa-user-shield me-1"></i>Admin Panel</a></li>'
        else:
            login_link = f'<li class="nav-item"><a class="nav-link fw-bold text-primary px-3" href="/doctor/dashboard"><i class="fas fa-user-md me-1"></i>Dashboard</a></li>'
        login_link += f'<li class="nav-item"><a class="nav-link text-danger px-2" href="/doctor/logout" title="Logout"><i class="fas fa-sign-out-alt"></i></a></li>'
    else:
        login_link = '<li class="nav-item"><a class="nav-link px-3" href="/doctor/login"><i class="fas fa-sign-in-alt me-1"></i>Staff Login</a></li>'

    language_selector = '''
                    <li class="nav-item ms-lg-3 notranslate">
                        <div class="language-selector d-flex align-items-center px-2 py-1">
                            <i class="fas fa-language text-primary me-2"></i>
                            <select class="form-select form-select-sm language-select" id="languageSwitcher" aria-label="Select language">
                                <option value="en">English</option>
                                <option value="kn">&#x0C95;&#x0CA8;&#xCCD;&#x0CA8;&#x0CA1;</option>
                                <option value="ml">&#x0D2E;&#x0D32;&#x0D2F;&#x0D3E;&#x0D33;&#x0D02;</option>
                                <option value="hi">&#x0939;&#x093F;&#x0902;&#x0926;&#x0940;</option>
                            </select>
                        </div>
                    </li>
    '''

    theme_selector = '''
                    <li class="nav-item ms-lg-2 notranslate">
                        <div class="theme-selector d-flex align-items-center px-2 py-1">
                            <i class="fas fa-circle-half-stroke text-primary me-2"></i>
                            <select class="form-select form-select-sm theme-select" id="themeSwitcher" aria-label="Select theme">
                                <option value="light">Light mode</option>
                                <option value="dark">Dark mode</option>
                            </select>
                        </div>
                    </li>
    '''

    nav = ''
    if show_nav:
        nav = f'''
    <!-- Professional Sticky Navigation Bar -->
    <nav id="mainNavbar" class="navbar navbar-expand-lg navbar-light bg-white shadow-sm sticky-top medflow-navbar">
        <div class="container">
            <a class="navbar-brand d-flex align-items-center notranslate" href="/">
                <div class="bg-primary text-white rounded p-1 me-2" style="width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;">
                    <i class="fas fa-hospital-alt"></i>
                </div>
                <div class="lh-1">
                    <div class="fw-bold brand-title" style="letter-spacing: 0.5px;">MEDFLOW</div>
                    <small class="brand-subtitle" style="font-size: 0.6rem;">SMART KIOSK FOR DIGITAL HEALTHCARE</small>
                </div>
            </a>
            <button class="navbar-toggler border-0 shadow-none" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto align-items-center">
                    <li class="nav-item"><a class="nav-link px-3" href="/"><i class="fas fa-th-large me-1"></i>Home</a></li>
                    <li class="nav-item"><a class="nav-link px-3" href="/register"><i class="fas fa-user-plus me-1"></i>Register</a></li>
                    <li class="nav-item"><a class="nav-link px-3" href="/queue"><i class="fas fa-list-ol me-1"></i>Queue</a></li>
                    <li class="nav-item"><a class="nav-link px-3" href="/navigation"><i class="fas fa-map-marked-alt me-1"></i>Map</a></li>
                    <li class="nav-item ms-lg-2"><a class="nav-link btn btn-danger btn-sm px-3 shadow-sm" href="/emergency"><i class="fas fa-ambulance me-1"></i>EMERGENCY</a></li>
                    <span class="mx-2 d-none d-lg-inline text-muted opacity-25">|</span>
                    {language_selector}
                    {theme_selector}
                    {login_link}
                </ul>
            </div>
        </div>
    </nav>
    '''
    
    # Get flash messages
    from flask import get_flashed_messages
    
    flash_messages = '''
    <div class="container mt-3">
    '''
    
    messages = get_flashed_messages(with_categories=True)
    if messages:
        flash_messages += '''
        <div class="row">
            <div class="col-12">
        '''
        for category, message in messages:
            alert_class = {
                'success': 'alert-success',
                'danger': 'alert-danger',
                'warning': 'alert-warning',
                'info': 'alert-info'
            }.get(category, 'alert-info')
            
            flash_messages += f'''
                <div class="alert {alert_class} alert-dismissible fade show flash-message" role="alert">
                    {message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>
            '''
        
        flash_messages += '''
            </div>
        </div>
        '''
    
    flash_messages += '''
    </div>
    '''
    
    # Chatbot HTML - FIXED VERSION
    chat_html = ''
    if show_chat:
        chat_html = '''
    <!-- Chatbot Widget - Fixed Position -->
    <div class="chatbot-container" id="chatbotContainer">
        <div class="chatbot-button" id="chatbotButton" onclick="toggleChat()">
            <i class="fas fa-comment-medical"></i>
            <span class="chat-notification" id="chatNotification">1</span>
        </div>
        
        <div class="chatbot-window" id="chatbotWindow">
            <div class="chatbot-header">
                <div class="d-flex align-items-center">
                    <i class="fas fa-robot me-2"></i>
                    <h6 class="mb-0">Registration Assistant</h6>
                </div>
                <button class="btn btn-sm text-white" onclick="toggleChat()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            
            <div class="chatbot-messages" id="chatMessages">
                <div class="message bot-message">
                    <div class="message-content">
                        <div class="d-flex align-items-start">
                            <div class="bot-icon me-2"><i class="fas fa-robot"></i></div>
                            <div>
                                <strong>Hello!</strong> I'm your registration assistant. I can help you with:
                                <ul class="mt-2 mb-0">
                                    <li>Registration process</li>
                                    <li>Aadhaar (12-digits)</li>
                                    <li>Face-capture check-in</li>
                                    <li>Queue status</li>
                                </ul>
                                <div class="mt-2">How can I help you today?</div>
                            </div>
                        </div>
                    </div>
                    <div class="message-time">Just now</div>
                </div>
            </div>
            
            <div class="chatbot-input">
                <div class="suggested-questions" id="suggestedQuestions">
                    <button class="suggested-question" onclick="sendSuggestedQuestion('How do I register?')">
                        How do I register?
                    </button>
                    <button class="suggested-question" onclick="sendSuggestedQuestion('What documents do I need?')">
                        What documents?
                    </button>
                    <button class="suggested-question" onclick="sendSuggestedQuestion('Which department should I visit?')">
                        Which department?
                    </button>
                    <button class="suggested-question" onclick="sendSuggestedQuestion('What is my queue number?')">
                        Queue number?
                    </button>
                </div>
                <div class="d-flex">
                    <input type="text" class="form-control" id="chatInput" placeholder="Type your question..." onkeypress="handleKeyPress(event)">
                    <button class="btn btn-primary ms-2" onclick="sendMessage()">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        </div>
    </div>
    '''
    
    # Chatbot JavaScript - FIXED VERSION
    chat_js = '''
    <script>
        // Chatbot functionality
        let chatSessionId = 'session_' + Date.now();
        let isTyping = false;
        
        function toggleChat() {
            try {
                const element = document.getElementById('chatbotWindow');
                if (element) {
                    const isNowShowing = element.style.display !== 'flex';
                    if (isNowShowing) {
                        element.style.cssText = 'display: flex !important; flex-direction: column !important;';
                        element.classList.add('show');
                        
                        // Hide notification when opened
                        const notif = document.getElementById('chatNotification');
                        if (notif) notif.classList.add('d-none');
                        
                        // Scroll to bottom
                        const msgs = document.getElementById('chatMessages');
                        if (msgs) msgs.scrollTop = msgs.scrollHeight;
                    } else {
                        element.style.cssText = 'display: none !important;';
                        element.classList.remove('show');
                    }
                }
            } catch(e) {
                console.error("Chat toggle:", e);
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        function sendSuggestedQuestion(question) {
            const input = document.getElementById('chatInput');
            if (input) {
                input.value = question;
                sendMessage();
            }
        }
        
        async function addMessage(message, sender) {
            const messagesDiv = document.getElementById('chatMessages');
            if (!messagesDiv) return;
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message notranslate`;
            messageDiv.setAttribute('translate', 'no');
            messageDiv.dataset.originalMessage = message;
            messageDiv.dataset.messageSender = sender;
            const targetLanguage = window.getHospitalLanguage ? window.getHospitalLanguage() : 'en';
            
            const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
            const translatedMessage = sender === 'bot' && window.translateHospitalTextAsync
                ? await window.translateHospitalTextAsync(message, targetLanguage)
                : message;

            // Basic Markdown-style formatting (**bold**)
            let formattedMessage = translatedMessage.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
            // Handle bullet points
            formattedMessage = formattedMessage.replace(/\\u2022\\s*(.*?)(?=\\n|$)/g, '<li>$1</li>');
            if (formattedMessage.includes('<li>')) {
                formattedMessage = formattedMessage.replace(/(<li>.*?<\\/li>)+/g, '<ul>$&</ul>');
            }
            
            messageDiv.innerHTML = `
                <div class="message-content">
                    <div class="d-flex align-items-start">
                        ${sender === 'bot' ? '<div class="bot-icon me-2"><i class="fas fa-robot"></i></div>' : ''}
                        <div class="message-body">${formattedMessage}</div>
                    </div>
                </div>
                <div class="message-time">${time}</div>
            `;
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
                if (window.refreshPageTranslation) {
                    window.refreshPageTranslation();
                }
        }
        
        function showTypingIndicator() {
            if (isTyping) return;
            isTyping = true;
            
            const messagesDiv = document.getElementById('chatMessages');
            if (!messagesDiv) return;
            
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message bot-message';
            typingDiv.id = 'typingIndicator';
            typingDiv.innerHTML = `
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            `;
            
            messagesDiv.appendChild(typingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function hideTypingIndicator() {
            const typingDiv = document.getElementById('typingIndicator');
            if (typingDiv) {
                typingDiv.remove();
            }
            isTyping = false;
        }
        
        function sendMessage() {
            const input = document.getElementById('chatInput');
            if (!input) return;
            
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage(message, 'user');
            input.value = '';
            
            // Show typing indicator
            showTypingIndicator();
            
            // Send to server
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: chatSessionId
                })
            })
            .then(response => response.json())
            .then(data => {
                hideTypingIndicator();
                
                // Add bot response
                if (data.responses && Array.isArray(data.responses)) {
                    data.responses.forEach(response => {
                        addMessage(response, 'bot');
                    });
                } else {
                    addMessage(data.response || "I'm here to help with registration!", 'bot');
                }
                
                // Show suggested next steps if available
                if (data.suggested_questions) {
                    updateSuggestedQuestions(data.suggested_questions);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                hideTypingIndicator();
                addMessage("Sorry, I'm having trouble connecting. Please try again.", 'bot');
            });
        }
        
        function updateSuggestedQuestions(questions) {
            const container = document.getElementById('suggestedQuestions');
            if (!container) return;
            
            container.innerHTML = '';
            questions.forEach(question => {
                const btn = document.createElement('button');
                btn.className = 'suggested-question';
                btn.textContent = question;
                btn.onclick = () => sendSuggestedQuestion(question);
                container.appendChild(btn);
            });
        }
        
        // Initialize chatbot when page loads
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Chatbot initialized');
            
            // Auto-resize textarea if needed
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.addEventListener('input', function() {
                    this.style.height = 'auto';
                    this.style.height = (this.scrollHeight) + 'px';
                });
            }
            
            // Handle window resize
            window.addEventListener('resize', function() {
                // Adjust chatbot position on mobile
                if (window.innerWidth <= 480) {
                    const container = document.querySelector('.chatbot-container');
                    if (container) {
                        container.style.bottom = '10px';
                        container.style.right = '10px';
                    }
                }
            });
            
            // Check for unread messages (simulated)
            setInterval(function() {
                const chatWindow = document.getElementById('chatbotWindow');
                const notification = document.getElementById('chatNotification');
                if (chatWindow && notification && !chatWindow.classList.contains('show')) {
                    // Simulate new message notification occasionally
                    if (Math.random() > 0.7) {
                        notification.classList.remove('d-none');
                    }
                }
            }, 30000);
        });
    </script>
    '''

    translation_js = '''
    <script src="/static/local_translate.js"></script>
    '''

    theme_init_script = '''
    <script>
        (function() {
            try {
                const storedTheme = window.localStorage.getItem('hospital_theme');
                const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                const theme = storedTheme === 'dark' || storedTheme === 'light'
                    ? storedTheme
                    : (prefersDark ? 'dark' : 'light');
                document.documentElement.setAttribute('data-theme', theme);
                document.documentElement.style.colorScheme = theme;
            } catch (error) {
                document.documentElement.setAttribute('data-theme', 'light');
                document.documentElement.style.colorScheme = 'light';
            }
        })();
    </script>
    '''
    
    # Responsive CSS - UPDATED with chatbot styles
    responsive_css = '''
    <style>
        /* Global Responsive Styles */
        :root {
            --primary-color: #3498db;
            --danger-color: #e74c3c;
            --success-color: #2ecc71;
            --warning-color: #f39c12;
            --dark-color: #2c3e50;
            --page-bg: #f8f9fa;
            --page-text: #1f2937;
            --surface-bg: #ffffff;
            --surface-alt: #f8fafc;
            --border-color: rgba(15, 23, 42, 0.08);
            --muted-text: #6b7280;
            --navbar-bg: rgba(255, 255, 255, 0.96);
            --navbar-text: #1f2937;
            --navbar-muted: #6b7280;
            --accent-color: #0d6efd;
            --footer-bg: #111827;
            --footer-text: #f8fafc;
            --footer-link: rgba(255, 255, 255, 0.78);
        }

        html[data-theme='dark'] {
            --page-bg: #0b1220;
            --page-text: #e5e7eb;
            --surface-bg: #111827;
            --surface-alt: #1f2937;
            --border-color: rgba(148, 163, 184, 0.18);
            --muted-text: #9ca3af;
            --navbar-bg: rgba(15, 23, 42, 0.96);
            --navbar-text: #e5e7eb;
            --navbar-muted: #94a3b8;
            --accent-color: #60a5fa;
            --footer-bg: #020617;
            --footer-text: #e5e7eb;
            --footer-link: rgba(226, 232, 240, 0.78);
            color-scheme: dark;
        }

        html[data-theme='light'] {
            color-scheme: light;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--page-bg);
            color: var(--page-text);
            overflow-x: hidden;
            min-height: 100vh;
            position: relative;
            top: 0 !important;
            padding-bottom: 60px; /* Space for footer */
            transition: background-color 0.25s ease, color 0.25s ease;
        }
        
        /* Responsive Typography */
        h1 { font-size: calc(1.5rem + 1.5vw); }
        h2 { font-size: calc(1.3rem + 1.2vw); }
        h3 { font-size: calc(1.2rem + 0.9vw); }
        h4 { font-size: calc(1.1rem + 0.6vw); }
        h5 { font-size: calc(1rem + 0.3vw); }
        p { font-size: calc(0.9rem + 0.1vw); }
        
        /* Responsive Cards */
        .card {
            border: none;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
            margin-bottom: 20px;
            height: 100%;
            background-color: var(--surface-bg);
            color: var(--page-text);
            border: 1px solid var(--border-color);
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }

        .medflow-navbar {
            background-color: var(--navbar-bg) !important;
            color: var(--navbar-text) !important;
            border-bottom: 1px solid var(--border-color);
            backdrop-filter: blur(12px);
        }

        .medflow-navbar .nav-link,
        .medflow-navbar .navbar-brand {
            color: var(--navbar-text) !important;
        }

        .medflow-navbar .brand-title {
            color: var(--accent-color) !important;
        }

        .medflow-navbar .brand-subtitle {
            color: var(--navbar-muted) !important;
        }

        .medflow-navbar .navbar-toggler {
            border-color: var(--border-color);
        }

        html[data-theme='dark'] .medflow-navbar .navbar-toggler-icon {
            filter: invert(1) brightness(1.4);
        }

        .navbar,
        .modal-content,
        .dropdown-menu,
        .chatbot-window,
        .form-control,
        .form-select,
        .input-group-text {
            background-color: var(--surface-bg);
            color: var(--page-text);
            border-color: var(--border-color);
        }

        .card-header,
        .card-footer,
        .modal-header,
        .modal-footer,
        .bg-white,
        .bg-light,
        .table-light,
        .list-group-item,
        .typing-indicator,
        .suggested-question {
            background-color: var(--surface-alt) !important;
            color: var(--page-text) !important;
            border-color: var(--border-color) !important;
        }

        .table {
            color: var(--page-text);
        }

        .text-muted {
            color: var(--muted-text) !important;
        }

        .language-selector,
        .theme-selector {
            min-width: 165px;
        }

        .language-select,
        .theme-select {
            min-width: 120px;
            font-size: 0.85rem;
            border-color: rgba(52, 152, 219, 0.35);
            box-shadow: none !important;
        }

        footer.bg-dark {
            background-color: var(--footer-bg) !important;
            color: var(--footer-text) !important;
        }

        footer.bg-dark a {
            color: var(--footer-link) !important;
        }
        
        /* Mobile Optimizations */
        @media (max-width: 768px) {
            .navbar-brand {
                font-size: 1.2rem;
            }

            .language-selector,
            .theme-selector {
                width: 100%;
                margin-top: 10px;
            }
            
            .btn {
                padding: 8px 16px;
                font-size: 0.9rem;
            }
            
            .table {
                font-size: 0.85rem;
            }
            
            .table td, .table th {
                padding: 0.5rem;
            }
            
            .doctor-dashboard {
                padding: 15px;
            }
            
            .card-body {
                padding: 1rem;
            }
            
            .department-icon {
                font-size: 2rem;
            }
            
            /* Stack buttons on mobile */
            .btn-group {
                flex-direction: column;
            }
            
            /* Smartphone Dashboard Optimizations */
            .admin-dashboard, .doctor-dashboard {
                padding: 15px;
            }
            
            .card.h-100 {
                margin-bottom: 10px;
            }
            
            .display-6 {
                font-size: 1.8rem;
            }
            
            /* Registration Form Mobile Polish */
            .face-capture-container {
                max-width: 100%;
                margin: 0 auto;
            }
            
            #camera-container {
                border-radius: 10px;
                overflow: hidden;
            }
            
            .btn-lg {
                padding: 0.8rem 1rem;
                font-size: 1rem;
            }
            
            /* Make tables scrollable on mobile */
            .table-responsive {
                border: 0;
                margin-bottom: 1rem;
                overflow-x: auto;
            }
            
            /* Emergency alert on mobile */
            .emergency-alert {
                padding: 0.75rem;
            }
            
            .emergency-alert .d-flex {
                flex-direction: column;
                text-align: center;
            }
            
            .emergency-alert i {
                margin-bottom: 10px;
                margin-right: 0 !important;
            }
        }
        
        /* Tablet Optimizations */
        @media (min-width: 769px) and (max-width: 1024px) {
            .card-columns {
                column-count: 2;
            }
            
            .container {
                max-width: 95%;
            }
        }
        
        /* Desktop Optimizations */
        @media (min-width: 1025px) {
            .container {
                max-width: 1200px;
            }
            
            .card-columns {
                column-count: 3;
            }
        }
        
        /* CHATBOT STYLES - FIXED AND VISIBLE */
        .chatbot-container {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 9999;
            display: block;
        }
        
        .chatbot-button {
            width: 70px;
            height: 70px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex !important;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            position: relative;
            border: 2px solid white;
            animation: pulse 2s infinite;
        }
        
        .chatbot-button:hover {
            transform: scale(1.1) rotate(5deg);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
        
        .chatbot-button i {
            font-size: 32px;
        }
        
        .chat-notification {
            position: absolute;
            top: -5px;
            right: -5px;
            background: var(--danger-color);
            color: white;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid white;
            font-weight: bold;
        }
        
        .chatbot-window {
            position: absolute;
            bottom: 90px;
            right: 0;
            width: 380px;
            height: 550px;
            background: var(--surface-bg);
            border-radius: 20px;
            display: none;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }
        
        @media (max-width: 480px) {
            .chatbot-window {
                width: 300px;
                height: 500px;
                right: -10px;
                bottom: 80px;
            }
            
            .chatbot-button {
                width: 60px;
                height: 60px;
            }
            
            .chatbot-button i {
                font-size: 28px;
            }
            
            .chatbot-container {
                bottom: 20px;
                right: 20px;
            }
        }
        
        .chatbot-window.show {
            display: flex !important;
        }
        
        .chatbot-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .chatbot-header h6 {
            font-size: 16px;
            font-weight: 600;
        }
        
        .chatbot-header button {
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .chatbot-header button:hover {
            background: rgba(255,255,255,0.3);
        }
        
        .chatbot-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: var(--page-bg);
        }
        
        .message {
            margin-bottom: 20px;
            max-width: 85%;
            clear: both;
        }
        
        .user-message {
            float: right;
        }
        
        .bot-message {
            float: left;
        }
        
        .message-content {
            padding: 12px 16px;
            border-radius: 18px;
            position: relative;
            word-wrap: break-word;
            line-height: 1.4;
        }
        
        .bot-message .message-content {
            background: var(--surface-bg);
            border: 1px solid var(--border-color);
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            font-size: 0.95rem;
        }

        .bot-icon {
            width: 28px;
            height: 28px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            flex-shrink: 0;
            margin-top: 2px;
        }
        
        .user-message .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
            font-size: 0.95rem;
        }
        
        .message-time {
            font-size: 10px;
            color: var(--muted-text);
            margin-top: 4px;
            padding: 0 4px;
        }
        
        .user-message .message-time {
            text-align: right;
            margin-right: 5px;
        }
        
        .chatbot-input {
            padding: 20px;
            background: var(--surface-bg);
            border-top: 1px solid var(--border-color);
        }
        
        .suggested-questions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
        }
        
        .suggested-question {
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 8px 15px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            white-space: nowrap;
            color: var(--page-text);
        }
        
        .suggested-question:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        @media (max-width: 480px) {
            .suggested-questions {
                overflow-x: auto;
                flex-wrap: nowrap;
                padding-bottom: 10px;
                -webkit-overflow-scrolling: touch;
            }
            
            .suggested-question {
                flex: 0 0 auto;
            }
        }
        
        .typing-indicator {
            display: flex;
            padding: 12px 16px;
            background: var(--surface-alt);
            border-radius: 18px;
            border: 1px solid var(--border-color);
            width: fit-content;
        }
        
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }
        
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
            70% { box-shadow: 0 0 0 15px rgba(102, 126, 234, 0); }
            100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
        }
        
        /* Utility Classes */
        .text-truncate-2 {
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        
        /* Loading Spinner */
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid var(--primary-color);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Status Badges */
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .status-waiting { background: var(--warning-color); color: white; }
        .status-in_progress { background: var(--primary-color); color: white; }
        .status-completed { background: var(--success-color); color: white; }
        .status-scheduled { background: #6c757d; color: white; }
        .status-emergency { background: var(--danger-color); color: white; animation: pulse 2s infinite; }
        
        /* Doctor Dashboard Styles */
        .doctor-dashboard {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .patient-status-waiting { border-left: 5px solid #f39c12; }
        .patient-status-in_progress { border-left: 5px solid #3498db; }
        .patient-status-completed { border-left: 5px solid #2ecc71; }
        .patient-status-scheduled { border-left: 5px solid #6c757d; }
        .patient-status-emergency { 
            border-left: 5px solid #e74c3c; 
            background-color: #ffeaea;
        }
        
        .btn-doctor {
            background-color: #9b59b6;
            border-color: #9b59b6;
            color: white;
        }
        
        .btn-doctor:hover {
            background-color: #8e44ad;
            color: white;
        }
        
        .btn-emergency {
            background-color: #e74c3c;
            border-color: #e74c3c;
            color: white;
        }
        
        .btn-emergency:hover {
            background-color: #c0392b;
            color: white;
        }
        
        .emergency-alert {
            animation: pulse 2s infinite;
            border-left: 5px solid #e74c3c;
        }
        
        /* Make sure chatbot is always on top */
        .chatbot-container {
            z-index: 9999;
        }

        .language-selector {
            min-width: 165px;
        }

        .language-select {
            min-width: 120px;
            font-size: 0.85rem;
            border-color: rgba(52, 152, 219, 0.35);
            box-shadow: none !important;
        }

    </style>
    '''
    
    return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=yes">
    <title>{title} - Smart Hospital Kiosk</title>
    {theme_init_script}
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    {responsive_css}
</head>
<body>
    {nav}
    
    <!-- Flash Messages -->
    {flash_messages}
    
    <!-- Main Content -->
    <div class="container mt-4">
        {content}
    </div>
    
    <!-- Professional Multi-Column Footer -->
    <footer class="bg-dark text-white pt-5 pb-4 mt-5">
        <div class="container text-center text-md-start">
            <div class="row">
                <!-- Column 1: Hospital Mission -->
                <div class="col-12 col-md-4 mb-4 mb-md-0">
                    <h5 class="text-uppercase fw-bold text-primary mb-3">
                        <i class="fas fa-hospital-alt me-2"></i>ＭＥＤＦＬＯＷ
                    </h5>
                    <p class="small text-muted mb-4" style="line-height: 1.8;">
                        Revolutionizing patient care with state-of-the-art digital self-service solutions. 
                        Streamlining hospital workflows for better health outcomes.
                    </p>
                    <div class="social-links">
                        <a href="#" class="text-white me-3 opacity-75 hover-opacity-100"><i class="fab fa-facebook-f"></i></a>
                        <a href="#" class="text-white me-3 opacity-75 hover-opacity-100"><i class="fab fa-twitter"></i></a>
                        <a href="#" class="text-white me-3 opacity-75 hover-opacity-100"><i class="fab fa-linkedin-in"></i></a>
                        <a href="#" class="text-white opacity-75 hover-opacity-100"><i class="fab fa-instagram"></i></a>
                    </div>
                </div>

                <!-- Column 2: Quick Navigation -->
                <div class="col-12 col-md-3 mb-4 mb-md-0">
                    <h6 class="text-uppercase fw-bold mb-4">Quick Links</h6>
                    <ul class="list-unstyled mb-0">
                        <li class="mb-2"><a href="/register" class="text-white-50 text-decoration-none hover-text-white">Patient Registration</a></li>
                        <li class="mb-2"><a href="/queue" class="text-white-50 text-decoration-none hover-text-white">Live Queue Status</a></li>
                        <li class="mb-2"><a href="/navigation" class="text-white-50 text-decoration-none hover-text-white">Internal Navigation</a></li>
                        <li class="mb-2"><a href="/emergency" class="text-white-50 text-decoration-none hover-text-white text-danger">Emergency Care</a></li>
                    </ul>
                </div>

                <!-- Column 3: Contact Info -->
                <div class="col-12 col-md-5">
                    <h6 class="text-uppercase fw-bold mb-4">Contact & Support</h6>
                    <p class="small text-white-50 mb-2"><i class="fas fa-map-marker-alt me-3 text-primary"></i> 123 Healthcare Blvd, Medical District</p>
                    <p class="small text-white-50 mb-2"><i class="fas fa-envelope me-3 text-primary"></i> medflowkiosk@support.com</p>
                    <p class="small text-white-50 mb-2"><i class="fas fa-phone me-3 text-primary"></i> +91 88123 56789</p>
                    <p class="small text-white-50 mb-0"><i class="fas fa-ambulance me-3 text-danger"></i> Emergency Hotline: <a href="tel:112" class="text-white-50 text-decoration-none hover-text-white"><strong>112</strong></a></p>
                    
                    <div class="mt-4">
                        <span class="badge bg-primary px-3 py-2">Open 24/7</span>
                        <span class="badge bg-outline-light text-white-50 border px-3 py-2 ms-2">ISO 9001 Certified</span>
                    </div>
                </div>
            </div>
            
            <hr class="my-4 opacity-25">
            
            <div class="row align-items-center pb-0">
                <div class="col-12 col-md-7 text-center text-md-start">
                    <p class="small text-white-50 mb-0">&copy; 2024 Smart Hospital Systems. All Rights Reserved.</p>
                </div>
                <div class="col-12 col-md-5 text-center text-md-end mt-3 mt-md-0">
                    <a href="#" class="small text-white-50 text-decoration-none me-3">Privacy Policy</a>
                    <a href="#" class="small text-white-50 text-decoration-none">Terms of Service</a>
                </div>
            </div>
        </div>
    </footer>
    
    <!-- Chatbot -->
    {chat_html}
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    {chat_js}
    {translation_js}
    
    <script>
        // Highlighting Active Navigation Link
        document.addEventListener("DOMContentLoaded", function() {{
            const currentPath = window.location.pathname;
            const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
            
            navLinks.forEach(link => {{
                const href = link.getAttribute('href');
                if (href === currentPath || (currentPath === '/' && href === '/')) {{
                    link.classList.add('active');
                    link.style.fontWeight = 'bold';
                    link.style.borderBottom = '2px solid var(--primary-color)';
                }}
            }});
        }});

        // Auto-refresh for queue page and doctor dashboard
        if (window.location.pathname === '/queue' || window.location.pathname.includes('/doctor/dashboard')) {{
            setInterval(() => {{
                location.reload();
            }}, 30000);
        }}
        
        // Auto-dismiss flash messages after 5 seconds
        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(function() {{
                const alerts = document.querySelectorAll('.flash-message');
                alerts.forEach(alert => {{
                    const bsAlert = new bootstrap.Alert(alert);
                    bsAlert.close();
                }});
            }}, 5000);
        }});
        
        // Touch-friendly hover effects for mobile
        if ('ontouchstart' in window) {{
            document.querySelectorAll('.card').forEach(card => {{
                card.addEventListener('touchstart', function() {{
                    this.style.transform = 'translateY(-5px)';
                }});
                card.addEventListener('touchend', function() {{
                    this.style.transform = 'translateY(0)';
                }});
            }});
        }}
        
        // Debug: Check if chatbot elements exist
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Checking chatbot elements...');
            const chatButton = document.getElementById('chatbotButton');
            const chatWindow = document.getElementById('chatbotWindow');
            console.log('Chat button exists:', !!chatButton);
            console.log('Chat window exists:', !!chatWindow);
            if (chatButton) {{
                console.log('Chat button styles:', window.getComputedStyle(chatButton).display);
            }}
        }});
    </script>
</body>
</html>
'''
# Doctor Login Decorator
def doctor_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role not in ['doctor', 'admin']:
            flash('Access denied. Doctor login required.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Admin Login Decorator
def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role != 'admin':
            flash('Access denied. Admin login required.', 'danger')
            return redirect(url_for('doctor_login'))
        return f(*args, **kwargs)
    return decorated_function

# Chatbot Routes
@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot messages"""
    data = request.json
    user_message = data.get('message', '').lower()
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    # Store user message
    user_msg = ChatMessage(
        session_id=session_id,
        sender='user',
        message=user_message
    )
    db.session.add(user_msg)
    
    # Process message and generate response
    responses = []
    intent = detect_intent(user_message)
    user_msg.intent = intent
    
    # Handle symptom-based department recommendations
    if intent == 'security_redact':
        responses = ["I am a patient assistant. For security reasons, I cannot provide information regarding Doctor or Admin login portals."]
    elif intent == 'face_alignment':
        responses = ["Our new Face Capture system uses **Eye-Based Alignment**. Please keep your head straight; the system will detect your eyes and automatically rotate the image to ensure high-accuracy identification."]
    elif intent.startswith('symptoms_'):
        dept_name = intent.split('_')[1]
        responses = get_symptom_based_department_info(dept_name)
    elif intent == 'registration':
        responses = get_registration_info()
    elif intent == 'documents':
        responses = get_documents_info()
    elif intent == 'department':
        responses = get_department_info(user_message)
    elif intent == 'queue':
        responses = get_queue_info(user_message)
    elif intent == 'emergency':
        responses = get_emergency_info()
    elif intent == 'greeting':
        responses = get_greeting_response()
    elif intent == 'cost':
        responses = get_cost_info()
    elif intent == 'timing':
        responses = get_timing_info()
    elif intent == 'location':
        responses = get_location_info(user_message)
    elif intent == 'contact':
        responses = get_contact_info()
    else:
        responses = get_fallback_response()
    
    # Store bot responses
    for response in responses:
        bot_msg = ChatMessage(
            session_id=session_id,
            sender='bot',
            message=response,
            intent=intent
        )
        db.session.add(bot_msg)
    
    db.session.commit()
    
    # Get suggested next questions
    suggested = get_suggested_questions(intent)
    
    return jsonify({
        'responses': responses,
        'intent': intent,
        'suggested_questions': suggested
    })

def detect_intent(message):
    """Detect the intent of the user message"""
    message = message.lower()
    
    # Security Block - Redact Doctor/Admin Dashboard data
    if any(word in message for word in ['admin', 'doctor login', 'password', 'login as doctor', 'dashboard access']):
        return 'security_redact'

    # Registration intent - Aadhaar & Face specific training
    if any(word in message for word in [
        'register', 'registration', 'sign up', 'enroll', 'appointment', 'book', 'new patient',
        'check in', 'admit', 'patient registration', 'make appointment', 'schedule visit',
        'consultation booking', 'doctor appointment', 'see doctor', 'visit doctor', 'how to register'
    ]):
        return 'registration'
    
    # Documents intent - Aadhaar specific training
    if any(word in message for word in [
        'document', 'documents', 'id', 'proof', 'card', 'paper', 'form', 'requirement',
        'required documents', 'what to bring', 'identification', 'aadhar', 'passport',
        'driving license', 'insurance card', 'medical records', 'previous reports',
        'referral letter', 'prescription', 'test reports', 'aadhar 12 digits'
    ]):
        return 'documents'
    
    # Face alignment training
    if any(word in message for word in ['face', 'photo', 'camera', 'capture', 'alignment', 'eye', 'accuracy']):
        return 'face_alignment'
    
    # Department intent - expanded keywords
    if any(word in message for word in [
        'department', 'departments', 'doctor', 'specialist', 'clinic', 'which department',
        'where to go', 'which doctor', 'specialty', 'cardiologist', 'gynecologist',
        'orthopedic', 'pediatrician', 'emergency doctor', 'pharmacy', 'pharmacist',
        'gynaecology', 'cardiology', 'orthopedics', 'pediatrics', 'emergency', 'pharmacy'
    ]):
        return 'department'
    
    # Queue intent - expanded keywords
    if any(word in message for word in [
        'queue', 'wait', 'waiting', 'status', 'how long', 'position', 'number', 'my turn',
        'queue number', 'waiting time', 'estimated time', 'how much longer', 'queue status',
        'check status', 'my position', 'queue position', 'waiting list'
    ]):
        return 'queue'
    
    # Emergency intent - expanded keywords
    if any(word in message for word in [
        'emergency', 'urgent', 'critical', 'accident', 'severe', 'immediate', 'help now',
        'life threatening', 'serious', 'ambulance', 'trauma', 'heart attack', 'stroke',
        'unconscious', 'bleeding', 'breathing difficulty', 'chest pain', 'severe pain'
    ]):
        return 'emergency'
    
    # Greeting intent - expanded keywords
    if any(word in message for word in [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'start', 'begin', 'help', 'assist', 'support', 'welcome', 'greetings'
    ]):
        return 'greeting'
    
    # Cost intent - expanded keywords
    if any(word in message for word in [
        'cost', 'price', 'fee', 'charge', 'payment', 'bill', 'expensive', 'how much',
        'consultation fee', 'registration fee', 'payment methods', 'insurance',
        'cash', 'card', 'upi', 'online payment', 'billing'
    ]):
        return 'cost'
    
    # Timing intent - expanded keywords
    if any(word in message for word in [
        'timing', 'timings', 'hour', 'hours', 'open', 'close', 'available', 'working time',
        'schedule', 'when open', 'operating hours', 'business hours', 'doctor hours',
        'appointment time', 'visiting hours', 'weekend hours'
    ]):
        return 'timing'
    
    # Location intent - expanded keywords
    if any(word in message for word in [
        'location', 'where', 'address', 'map', 'direction', 'directions', 'find', 'reach',
        'how to get', 'navigation', 'floor', 'building', 'parking', 'nearby', 'distance'
    ]):
        return 'location'
    
    # Contact intent - expanded keywords
    if any(word in message for word in [
        'contact', 'phone', 'call', 'email', 'reach', 'support', 'helpline',
        'phone number', 'contact number', 'hotline', 'customer care', 'feedback'
    ]):
        return 'contact'
    
    # Check for specific department symptoms
    symptom_keywords = {
        'cardiology': ['heart', 'chest', 'bp', 'blood pressure', 'cardiac', 'palpitations', 'angina', 'hypertension', 'cholesterol'],
        'gynaecology': ['pregnancy', 'women', 'gynecologist', 'menstrual', 'womb', 'uterus', 'ovary', 'period', 'pregnant', 'gyne'],
        'orthopedics': ['bone', 'joint', 'fracture', 'sprain', 'muscle', 'back pain', 'knee', 'shoulder', 'arm', 'leg', 'spine'],
        'pediatrics': ['child', 'baby', 'infant', 'pediatric', 'kids', 'children', 'fever', 'cough', 'vaccination', 'growth'],
        'emergency': ['accident', 'injury', 'bleeding', 'unconscious', 'severe pain', 'difficulty breathing', 'chest pain']
    }
    
    for dept, keywords in symptom_keywords.items():
        if any(word in message for word in keywords):
            return 'symptoms_' + dept
    
    return 'unknown'

def get_registration_info():
    """Get accurate registration information based on actual webapp"""
    return [
        "📝 **Patient Registration Process:**\n\n" +
        "1. Click on 'Register' in the navigation menu\n" +
        "2. Fill in your personal details:\n" +
        "   • Full name (required)\n" +
        "   • Age (required)\n" +
        "   • Gender (Male/Female/Other)\n" +
        "   • Phone number (required)\n" +
        "   • Email address (optional)\n" +
        "   • Aadhaar number (**Strict 12 digits, No spaces/hyphens**)\n" +
        "3. Select your department from available options\n" +
        "4. Choose consultation type (General/Specialist)\n" +
        "5. Choose your consultation date\n" +
        "6. **Face Registration**: Our system now uses eye-detection to align your photo automatically!\n" +
        "7. Submit the form. The system predicts your consultation time automatically from the queue.",
        
        "After registration, you'll receive:\n" +
        "✅ Unique Patient ID (e.g., PAT12345678)\n" +
        "✅ Queue Number (e.g., CAR001)\n" +
        "✅ Department assignment\n" +
        "✅ Estimated consultation time\n" +
        "✅ Login credentials (Patient ID as username/password)"
    ]

def get_documents_info():
    """Get documents information including Aadhaar"""
    return [
        "📋 **Required Documents:**\n\n" +
        "• Government ID (Aadhar, Passport, Driver's License)\n" +
        "• Previous medical records (if any)\n" +
        "• Insurance card (if applicable)\n" +
        "• Referral letter (if referred by another doctor)",
        
        "**Optional but Recommended:**\n" +
        "• Aadhaar number (can be entered during registration)\n" +
        "• Recent medical reports\n" +
        "• Prescription copies",
        
        "For follow-up visits, please bring:\n" +
        "• Previous prescription\n" +
        "• Test reports\n" +
        "• Your Patient ID card"
    ]

def get_department_info(message):
    """Get department information based on symptoms"""
    message = message.lower()
    
    # Simple symptom-based department suggestions
    if any(word in message for word in ['heart', 'chest', 'bp', 'blood pressure', 'cardiac']):
        return [
            "❤️ Based on your symptoms, you should visit the **Cardiology Department**.",
            "Location: Main Building, Second Floor\n" +
            "Available Doctors: Dr. John Smith, Dr. Robert Miller, Dr. Lisa Chen\n" +
            "Would you like me to help you register for Cardiology?"
        ]
    elif any(word in message for word in ['bone', 'joint', 'fracture', 'sprain', 'muscle', 'back pain']):
        return [
            "🦴 Based on your symptoms, you should visit the **Orthopedics Department**.",
            "Location: Main Building, Second Floor\n" +
            "Available Doctors: Dr. Raj Patel, Dr. Carlos Rodriguez\n" +
            "Would you like me to help you register for Orthopedics?"
        ]
    elif any(word in message for word in ['pregnancy', 'women', 'gynecologist', 'menstrual', 'womb']):
        return [
            "👩 Based on your symptoms, you should visit the **Gynaecology Department**.",
            "Location: Main Building, First Floor\n" +
            "Available Doctors: Dr. Emily Wang, Dr. Priya Sharma, Dr. Maria Jackson\n" +
            "Would you like me to help you register for Gynaecology?"
        ]
    elif any(word in message for word in ['child', 'baby', 'infant', 'pediatric', 'kids']):
        return [
            "👶 Based on your symptoms, you should visit the **Pediatrics Department**.",
            "Location: Main Building, First Floor\n" +
            "Available Doctors: Dr. David Anderson, Dr. Anjali Gupta\n" +
            "Would you like me to help you register for Pediatrics?"
        ]
    else:
        # Get all departments
        departments = Department.query.all()
        dept_list = "\n".join([f"• {dept.name}: {dept.description}" for dept in departments])
        
        return [
            "🏥 **Available Departments:**\n\n" + dept_list,
            "Please describe your symptoms, and I can recommend the most suitable department, or you can select one during registration."
        ]

def get_symptom_based_department_info(dept_name):
    """Get department information based on specific symptoms"""
    dept_info = {
        'cardiology': {
            'name': 'Cardiology',
            'icon': '❤️',
            'symptoms': ['chest pain', 'heart palpitations', 'shortness of breath', 'high blood pressure', 'cholesterol issues'],
            'doctors': ['Dr. John Smith', 'Dr. Robert Miller', 'Dr. Lisa Chen'],
            'location': 'Main Building, Second Floor'
        },
        'gynaecology': {
            'name': 'Gynaecology',
            'icon': '👩',
            'symptoms': ['pregnancy concerns', 'menstrual issues', 'uterine problems', 'ovarian issues', 'women\'s health'],
            'doctors': ['Dr. Emily Wang', 'Dr. Priya Sharma', 'Dr. Maria Jackson'],
            'location': 'Main Building, First Floor'
        },
        'orthopedics': {
            'name': 'Orthopedics',
            'icon': '🦴',
            'symptoms': ['bone fractures', 'joint pain', 'muscle injuries', 'back pain', 'sports injuries'],
            'doctors': ['Dr. Raj Patel', 'Dr. Carlos Rodriguez'],
            'location': 'Main Building, Second Floor'
        },
        'pediatrics': {
            'name': 'Pediatrics',
            'icon': '👶',
            'symptoms': ['child fever', 'cough', 'vaccination', 'growth concerns', 'child health issues'],
            'doctors': ['Dr. David Anderson', 'Dr. Anjali Gupta'],
            'location': 'Main Building, First Floor'
        },
        'emergency': {
            'name': 'Emergency',
            'icon': '🚨',
            'symptoms': ['severe injury', 'unconsciousness', 'heavy bleeding', 'difficulty breathing', 'chest pain'],
            'doctors': ['Dr. Sarah Jones', 'Dr. Rajesh Kumar'],
            'location': 'Main Building, Ground Floor'
        }
    }
    
    if dept_name in dept_info:
        info = dept_info[dept_name]
        symptoms_list = "\n• ".join(info['symptoms'])
        doctors_list = "\n• ".join(info['doctors'])
        
        return [
            f"{info['icon']} **{info['name']} Department Recommended**\n\n" +
            f"**Common symptoms we treat:**\n• {symptoms_list}\n\n" +
            f"**Available Doctors:**\n• {doctors_list}\n\n" +
            f"**Location:** {info['location']}",
            
            f"Would you like me to help you register for the {info['name']} Department? " +
            "You can also proceed directly to the department if this is urgent."
        ]
    
    return get_fallback_response()

def get_queue_info(message):
    """Get queue information"""
    # Try to extract patient ID or name from message
    import re
    patient_id_match = re.search(r'[A-Z]{3}\d{3}', message.upper())
    
    if patient_id_match:
        patient_id = patient_id_match.group()
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        
        if patient:
            if patient.status == 'scheduled':
                return [
                    f"📅 **Appointment Status for {patient.name}**\n\n" +
                    f"Patient ID: {patient.patient_id}\n" +
                    f"Department: {patient.department}\n" +
                    f"Status: {patient.status.upper()}\n" +
                    f"Consultation Date: {format_consultation_date(patient.consultation_date)}\n" +
                    f"Estimated Consultation Time: {format_consultation_time(patient.consultation_time)}\n" +
                    "Queue Number: Will be assigned on arrival"
                ]

            # Get queue position
            queue_position = Patient.query.filter(
                Patient.department == patient.department,
                Patient.status.in_(['waiting', 'emergency']),
                Patient.check_in_time < patient.check_in_time
            ).count() + 1
            
            wait_time = queue_position * get_consultation_slot_minutes()
            
            return [
                f"📊 **Queue Status for {patient.name}**\n\n" +
                f"Patient ID: {patient.patient_id}\n" +
                f"Queue Number: {patient.queue_number}\n" +
                f"Department: {patient.department}\n" +
                f"Status: {patient.status.upper()}\n" +
                f"Position in Queue: {queue_position}\n" +
                f"Estimated Consultation Time: {format_consultation_time(patient.consultation_time)}\n" +
                f"Estimated Wait Time: {wait_time} minutes"
            ]
    
    # Get general queue status by department
    departments = Department.query.all()
    queue_status = "**Current Queue Status:**\n\n"
    
    for dept in departments:
        waiting = Patient.query.filter_by(department=dept.name, status='waiting').count()
        emergency = Patient.query.filter_by(department=dept.name, status='emergency').count()
        in_progress = Patient.query.filter_by(department=dept.name, status='in_progress').count()
        
        if waiting > 0 or emergency > 0:
            queue_status += f"**{dept.name}:**\n"
            if emergency > 0:
                queue_status += f"🚨 Emergency: {emergency}\n"
            queue_status += f"⏳ Waiting: {waiting}\n"
            queue_status += f"👨‍⚕️ In Progress: {in_progress}\n\n"
    
    return [queue_status, "To check your personal queue status, please provide your Patient ID (e.g., PAT12345678)"]

def get_emergency_info():
    """Get emergency information"""
    return [
        "🚨 **EMERGENCY ASSISTANCE** 🚨\n\n" +
        "If this is a medical emergency:\n\n" +
        "📞 **CALL 112 IMMEDIATELY**\n\n" +
        "Or proceed directly to:\n" +
        "🏥 Emergency Department\n" +
        "📍 Main Building, Ground Floor\n" +
        "🕐 Open 24/7",
        
        "Emergency services available:\n" +
        "• Ambulance services\n" +
        "• Trauma care\n" +
        "• Critical care unit\n" +
        "• Emergency surgery\n" +
        "• 24/7 specialist doctors"
    ]

def get_greeting_response():
    """Get enhanced greeting response with webapp features"""
    return [
        "🏥 **Welcome to Smart Hospital Kiosk!**\n\n" +
        "Hello! I'm your AI Hospital Assistant, here to help you navigate our services.",
        
        "**I can assist you with:**\n" +
        "• 📝 **Patient Registration** - Quick and easy online registration\n" +
        "• 📋 **Document Requirements** - Know what to bring\n" +
        "• 🏥 **Department Selection** - Find the right specialist\n" +
        "• ⏳ **Queue Management** - Check your waiting status\n" +
        "• 🚨 **Emergency Services** - 24/7 emergency care\n" +
        "• 💰 **Cost Information** - Transparent pricing\n" +
        "• 🕐 **Hospital Timings** - When we're open\n" +
        "• 📍 **Navigation** - Find your way around\n" +
        "• 📞 **Contact Information** - How to reach us",
        
        "**Quick Actions:**\n" +
        "• Register as a new patient\n" +
        "• Check queue status\n" +
        "• Find department locations\n" +
        "• Emergency assistance",
        
        "What would you like help with today?"
    ]

def get_cost_info():
    """Get cost information"""
    return [
        "💰 **Consultation Fees:**\n\n" +
        "• New Patient Registration: ₹500\n" +
        "• Follow-up Visit: ₹300\n" +
        "• Emergency Consultation: ₹1000\n" +
        "• Specialist Consultation: ₹800",
        
        "Additional costs may apply for:\n" +
        "• Diagnostic tests\n" +
        "• Medications\n" +
        "• Procedures\n" +
        "• Hospital stay",
        
        "We accept:\n" +
        "• Cash\n" +
        "• Credit/Debit cards\n" +
        "• UPI (Google Pay, PhonePe, Paytm)\n" +
        "• Health insurance"
    ]

def get_timing_info():
    """Get timing information"""
    return [
        "🕐 **Hospital Timings:**\n\n" +
        "• OPD: Monday to Saturday, 9:00 AM - 5:00 PM\n" +
        "• Emergency: 24/7, all days\n" +
        "• Pharmacy: 8:00 AM - 8:00 PM\n" +
        "• Laboratory: 7:00 AM - 7:00 PM",
        
        "**Doctor Availability:**\n" +
        "• General Physicians: 9 AM - 5 PM\n" +
        "• Specialists: By appointment\n" +
        "• Emergency Doctors: 24/7"
    ]

def get_location_info(message):
    """Get location information"""
    # Check if specific department mentioned
    departments = Department.query.all()
    for dept in departments:
        if dept.name.lower() in message.lower():
            return [
                f"📍 **{dept.name} Department Location:**\n\n" +
                f"Location: {dept.location}\n" +
                f"Floor: {dept.floor}\n" +
                f"Description: {dept.description}",
                
                f"🅿️ Parking available nearby\n" +
                f"♿ Wheelchair accessible\n" +
                f"🚻 Restrooms nearby"
            ]
    
    # General location info
    return [
        "🏥 **Hospital Location:**\n\n" +
        "Smart Hospital\n" +
        "123 Healthcare Avenue\n" +
        "Medical District\n" +
        "City - 400001",
        
        "**How to Reach:**\n" +
        "🚇 Metro: Medical College Station (5 min walk)\n" +
        "🚌 Bus: Route 123, 456 (Hospital Stop)\n" +
        "🚗 Taxi: Available 24/7\n" +
        "🅿️ Parking: Basement and ground floor",
        
        "Use the Navigation page for detailed floor-wise maps."
    ]

def get_contact_info():
    """Get contact information"""
    return [
        "📞 **Contact Us:**\n\n" +
        "• Main Reception: +91 12345 67890\n" +
        "• Emergency: 112\n" +
        "• Ambulance: 108\n" +
        "• Appointment: +91 98765 43210\n" +
        "• Email: info@smarthospital.com",
        
        "📱 **Social Media:**\n" +
        "• Facebook: /smarthospital\n" +
        "• Twitter: @smarthospital\n" +
        "• Instagram: @smarthospital"
    ]

def get_fallback_response():
    """Get enhanced fallback response with better guidance"""
    return [
        "🤔 **I didn't quite understand that.**\n\n" +
        "I'm here to help with hospital services. Let me assist you with:",
        
        "**Common Questions:**\n" +
        "• How do I register as a patient?\n" +
        "• What documents do I need to bring?\n" +
        "• Which department should I visit?\n" +
        "• How can I check my queue status?\n" +
        "• What are the hospital timings?\n" +
        "• How do I reach the hospital?\n" +
        "• Emergency contact information",
        
        "**For specific symptoms, try:**\n" +
        "• 'I have chest pain' → Cardiology\n" +
        "• 'Pregnancy related' → Gynaecology\n" +
        "• 'Broken bone' → Orthopedics\n" +
        "• 'Child fever' → Pediatrics\n" +
        "• 'Severe injury' → Emergency",
        
        "Please rephrase your question, or choose from the suggested questions below. " +
        "You can also describe your symptoms for department recommendations!"
    ]

def get_suggested_questions(intent):
    """Get enhanced suggested next questions based on intent"""
    suggestions = {
        'registration': [
            'What documents do I need?',
            'How much does registration cost?',
            'Which department should I choose?',
            'What is the registration process?',
            'Can I register online?'
        ],
        'documents': [
            'How do I register?',
            'What if I forgot my ID?',
            'Do I need insurance?',
            'Can I bring previous medical records?',
            'What ID proofs are accepted?'
        ],
        'department': [
            'Show all departments',
            'Cardiology department',
            'Emergency department',
            'Gynaecology department',
            'Orthopedics department',
            'Pediatrics department'
        ],
        'queue': [
            'Check my queue status',
            'How long is the wait?',
            'Emergency queue',
            'Cancel my appointment',
            'Change my department'
        ],
        'emergency': [
            'Emergency contact number',
            'Ambulance service',
            'Emergency department location',
            'Trauma care available',
            'Emergency doctors available'
        ],
        'cost': [
            'Insurance accepted?',
            'Payment methods available',
            'Follow-up visit cost',
            'Emergency consultation fee',
            'Specialist consultation cost'
        ],
        'timing': [
            'Weekend hours',
            'Doctor availability',
            'Emergency hours',
            'Pharmacy timings',
            'Laboratory hours'
        ],
        'location': [
            'How to reach the hospital?',
            'Parking facility available?',
            'Department locations',
            'Emergency entrance',
            'Pharmacy location'
        ],
        'contact': [
            'Phone numbers',
            'Email support',
            'Emergency contact',
            'Social media',
            'Feedback and complaints'
        ],
        'greeting': [
            'How do I register?',
            'What documents do I need?',
            'Which department?',
            'Emergency help',
            'Hospital timings'
        ],
        'symptoms_cardiology': [
            'Cardiology registration',
            'Heart specialists available',
            'Cardiology timings',
            'Emergency heart care',
            'Cardiology location'
        ],
        'symptoms_gynaecology': [
            'Gynaecology registration',
            'Women health specialists',
            'Gynaecology timings',
            'Pregnancy care',
            'Gynaecology location'
        ],
        'symptoms_orthopedics': [
            'Orthopedics registration',
            'Bone specialists available',
            'Orthopedics timings',
            'Emergency bone care',
            'Orthopedics location'
        ],
        'symptoms_pediatrics': [
            'Pediatrics registration',
            'Child specialists available',
            'Pediatrics timings',
            'Emergency child care',
            'Pediatrics location'
        ],
        'symptoms_emergency': [
            'Emergency registration',
            'Emergency doctors available',
            'Emergency timings',
            'Trauma care',
            'Emergency location'
        ],
        'unknown': [
            'How do I register?',
            'What documents do I need?',
            'Hospital timings',
            'Emergency contact',
            'Department information'
        ]
    }
    
    return suggestions.get(intent, suggestions['unknown'])

# Routes
@app.route('/')
def index():
    """Home page - main entry point"""
    departments = Department.query.all()
    
    # Get all doctors grouped by department
    doctors_by_dept = {}
    doctors = User.query.filter_by(role='doctor').all()
    for doctor in doctors:
        if doctor.department not in doctors_by_dept:
            doctors_by_dept[doctor.department] = []
        doctors_by_dept[doctor.department].append({
            'name': doctor.full_name,
            'id': doctor.id
        })
    
    # Get queue statistics
    total_waiting = Patient.query.filter_by(status='waiting').count()
    total_emergency = Patient.query.filter_by(status='emergency').count()
    total_in_progress = Patient.query.filter_by(status='in_progress').count()
    
    content = f'''
    <!-- Hero Section - Responsive -->
    <div class="row align-items-center py-3 py-md-5">
        <div class="col-12 col-lg-6 text-center text-lg-start mb-4 mb-lg-0">
            <h1 class="display-4 fw-bold mb-3">Welcome to Smart Hospital Kiosk</h1>
            <p class="lead mb-4">Streamline your hospital visit with our intelligent self-service system.</p>
            
            <div class="d-grid gap-2 d-md-flex justify-content-center justify-content-lg-start mb-4">
                <a href="/register" class="btn btn-primary btn-lg px-4 me-md-2">
                    <i class="fas fa-user-plus me-2"></i>Register Now
                </a>
                <a href="/queue" class="btn btn-outline-primary btn-lg px-4">
                    <i class="fas fa-list-ol me-2"></i>Check Queue
                </a>
                <a href="/navigation" class="btn btn-outline-primary btn-lg px-4">
                    <i class="fas fa-map-marked-alt me-1"></i>Map
                </a>
            </div>
            
            <!-- Statistics Cards - Responsive -->
            <div class="row g-2 g-md-3 mb-4">
                <div class="col-4">
                    <div class="card bg-warning text-white text-center p-2 p-md-3">
                        <i class="fas fa-clock fs-1 mb-2"></i>
                        <h3 class="mb-0">{total_waiting}</h3>
                        <small>Waiting</small>
                    </div>
                </div>
                <div class="col-4">
                    <div class="card bg-danger text-white text-center p-2 p-md-3">
                        <i class="fas fa-ambulance fs-1 mb-2"></i>
                        <h3 class="mb-0">{total_emergency}</h3>
                        <small>Emergency</small>
                    </div>
                </div>
                <div class="col-4">
                    <div class="card bg-info text-white text-center p-2 p-md-3">
                        <i class="fas fa-user-md fs-1 mb-2"></i>
                        <h3 class="mb-0">{total_in_progress}</h3>
                        <small>In Progress</small>
                    </div>
                </div>
            </div>
            
            <!-- Emergency Alert - Responsive -->
            <div class="alert alert-danger emergency-alert" role="alert">
                <div class="d-flex flex-column flex-md-row align-items-center">
                    <i class="fas fa-ambulance fa-2x me-md-3 mb-2 mb-md-0"></i>
                    <div class="text-center text-md-start">
                        <h5 class="alert-heading mb-1">Emergency?</h5>
                        <p class="mb-0">Proceed directly to Emergency Department or call <a href="tel:112" class="fw-bold text-decoration-none" style="color: inherit;">112</a></p>
                    </div>
                </div>
                <div class="mt-3 text-center">
                    <a href="/emergency" class="btn btn-danger w-100 w-md-auto">
                        <i class="fas fa-exclamation-triangle me-2"></i>Emergency Assistance
                    </a>
                </div>
            </div>
        </div>
        
        <div class="col-12 col-lg-6 text-center d-none d-lg-block">
            <i class="fas fa-hospital fa-10x text-primary"></i>
        </div>
    </div>
    
    <!-- Departments Section - Responsive Grid -->
    <div class="row py-3 py-md-5">
        <div class="col-12">
            <h2 class="text-center mb-4">Our Departments</h2>
            <p class="text-center text-muted mb-4">Click on any department to see available doctors</p>
        </div>
    '''
    
    for dept in departments:
        doctors_list = doctors_by_dept.get(dept.name, [])
        doctors_count = len(doctors_list)
        
        # Create doctors list HTML
        doctors_html = ''
        for doctor in doctors_list:
            doctors_html += f'''
            <div class="doctor-card p-2 p-md-3">
                <div class="d-flex align-items-center">
                    <div class="me-2 me-md-3">
                        <i class="fas fa-user-md fa-2x text-primary"></i>
                    </div>
                    <div class="flex-grow-1">
                        <h6 class="mb-1">{doctor['name']}</h6>
                        <p class="small text-muted mb-0">{dept.name} Department</p>
                    </div>
                    <div class="text-end">
                        <span class="badge bg-success">Available</span>
                    </div>
                </div>
            </div>
            '''
        
        if not doctors_html:
            doctors_html = '''
            <div class="text-center py-3">
                <i class="fas fa-user-md fa-3x text-muted mb-3"></i>
                <h6>No doctors available</h6>
                <p class="text-muted small">Currently no doctors are available in this department.</p>
            </div>
            '''
        
        content += f'''
        <div class="col-12 col-sm-6 col-lg-4 mb-4">
            <div class="card h-100" style="cursor: pointer;" data-bs-toggle="modal" data-bs-target="#deptModal{dept.id}">
                <div class="card-body text-center">
                    <div class="department-icon">
                        <i class="{dept.icon}"></i>
                    </div>
                    <h5 class="card-title">{dept.name}</h5>
                    <p class="card-text text-muted small text-truncate-2">{dept.description}</p>
                    
                    <div class="mt-3">
                        <span class="badge bg-primary">
                            <i class="fas fa-user-md me-1"></i>
                            {doctors_count} Doctor{'' if doctors_count == 1 else 's'} Available
                        </span>
                    </div>
                    
                    <div class="mt-2">
                        <span class="badge bg-light text-dark">
                            <i class="fas fa-map-marker-alt me-1"></i>
                            {dept.location}
                        </span>
                    </div>
                    
                    <div class="mt-3">
                        <button class="btn btn-sm btn-outline-primary w-100" data-bs-toggle="modal" data-bs-target="#deptModal{dept.id}">
                            <i class="fas fa-stethoscope me-1"></i>View Doctors
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Modal for {dept.name} - Responsive -->
        <div class="modal fade" id="deptModal{dept.id}" tabindex="-1" aria-labelledby="deptModalLabel{dept.id}" aria-hidden="true">
            <div class="modal-dialog modal-dialog-centered modal-dialog-scrollable">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="deptModalLabel{dept.id}">{dept.name} - Available Doctors</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="text-start">
                            <h6>Available Doctors ({doctors_count}):</h6>
                            {doctors_html}
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <a href="/register?department={dept.name}" class="btn btn-primary">
                            <i class="fas fa-user-plus me-2"></i>Register Here
                        </a>
                    </div>
                </div>
            </div>
        </div>
        '''
    
    content += '''
    </div>
    
    <!-- Quick Links for Mobile -->
    <div class="row d-md-none mt-4">
        <div class="col-12">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Quick Actions</h5>
                    <div class="d-grid gap-2">
                        <a href="/register" class="btn btn-primary">
                            <i class="fas fa-user-plus me-2"></i>New Registration
                        </a>
                        <a href="/queue" class="btn btn-outline-primary">
                            <i class="fas fa-list-ol me-2"></i>Check Queue
                        </a>
                        <a href="/navigation" class="btn btn-outline-primary">
                            <i class="fas fa-map-signs me-2"></i>Hospital Map
                        </a>
                        <a href="/emergency" class="btn btn-danger">
                            <i class="fas fa-ambulance me-2"></i>Emergency
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <style>
        .doctor-card {
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            transition: all 0.3s ease;
        }
        
        .doctor-card:hover {
            background-color: #f8f9fa;
            border-color: #3498db;
            transform: translateX(5px);
        }
        
        @media (max-width: 768px) {
            .doctor-card {
                padding: 10px;
            }
            
            .doctor-card h6 {
                font-size: 0.9rem;
            }
            
            .doctor-card .badge {
                font-size: 0.7rem;
            }
        }
    </style>
    '''
    
    return get_base_html("Home", content, show_chat=True)

@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    """Doctor login page"""
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            if user.role == 'doctor':
                login_user(user)
                return redirect(url_for('doctor_dashboard'))
            elif user.role == 'admin':
                login_user(user)
                return redirect(url_for('admin_dashboard'))
            else:
                error = '<div class="alert alert-danger">Access denied. User role not authorized for this portal.</div>'
        else:
            error = '<div class="alert alert-danger">Invalid username or password</div>'
    else:
        error = ''
    
    content = f'''
    <div class="row justify-content-center">
        <div class="col-12 col-md-8 col-lg-6">
            <div class="card">
                <div class="card-body p-3 p-md-4">
                    <h2 class="text-center mb-4"><i class="fas fa-user-md me-2"></i>Doctor Login</h2>
                    {error}
                    <form method="POST">
                        <div class="mb-3">
                            <label class="form-label">Username</label>
                            <input type="text" class="form-control" name="username" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Password</label>
                            <input type="password" class="form-control" name="password" required>
                        </div>
                        <div class="d-grid">
                            <button type="submit" class="btn btn-doctor">Login</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
    '''
    
    return get_base_html("Doctor Login", content, show_chat=False)

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard for overall hospital management"""
    admin_name = current_user.full_name
    
    # Hospital Stats
    total_patients = Patient.query.count()
    waiting_patients = Patient.query.filter_by(status='waiting').count()
    emergency_patients = Patient.query.filter_by(status='emergency').count()
    doctors = User.query.filter_by(role='doctor').all()
    departments = Department.query.order_by(Department.name).all()

    department_options = ''
    for dept in departments:
        department_name = html.escape(dept.name or '')
        department_options += f'<option value="{department_name}">{department_name}</option>'
    if not department_options:
        department_options = '<option value="" selected disabled>No departments available</option>'

    doctor_rows = ''
    for doctor in doctors:
        doctor_account_id = f'#{doctor.id}'
        doctor_name = html.escape(doctor.full_name or doctor.username or 'Doctor')
        doctor_username = html.escape(doctor.username or '')
        doctor_department = html.escape(doctor.department or '-')
        doctor_confirm_name = (doctor.full_name or doctor.username or 'this doctor').replace('\\', '\\\\').replace("'", "\\'")
        doctor_rows += f'''
                            <tr>
                                <td><span class="badge bg-light text-dark border">{doctor_account_id}</span></td>
                                <td>
                                    <div class="fw-bold">{doctor_name}</div>
                                </td>
                                <td>
                                    <code class="text-primary">{doctor_username}</code>
                                </td>
                                <td>{doctor_department}</td>
                                <td>
                                    <span class="badge bg-success">Active</span>
                                </td>
                                <td class="text-end">
                                    <form method="POST" action="/admin/doctors/{doctor.id}/delete" class="d-inline" onsubmit="return confirm('Remove {doctor_confirm_name} from the hospital system?');">
                                        <button type="submit" class="btn btn-outline-danger btn-sm">
                                            <i class="fas fa-trash me-1"></i>Remove
                                        </button>
                                    </form>
                                </td>
                            </tr>
        '''
    if not doctor_rows:
        doctor_rows = '''
                            <tr>
                                <td colspan="6" class="text-center text-muted py-4">
                                    No doctors found. Add one using the form on the left.
                                </td>
                            </tr>
        '''
    
    # Recent Patients
    recent_patients = Patient.query.order_by(Patient.check_in_time.desc()).limit(10).all()
    
    content = f'''
    <div class="admin-dashboard">
        <div class="row align-items-center">
            <div class="col-12 col-md-8">
                <h1 class="display-5 fw-bold mb-3"><i class="fas fa-user-shield me-2"></i>Admin Dashboard</h1>
                <h3>Welcome, {admin_name}</h3>
                <p class="text-muted">Hospital Wide Management System</p>
            </div>
            <div class="col-12 col-md-4 text-md-end">
                <a href="/doctor/logout" class="btn btn-outline-danger">
                    <i class="fas fa-sign-out-alt me-2"></i>Logout
                </a>
            </div>
        </div>

        <div class="row g-4 mt-2">
            <div class="col-12 col-sm-6 col-lg-3">
                <div class="card bg-primary text-white h-100">
                    <div class="card-body">
                        <h5>Total Patients</h5>
                        <h2 class="display-6 fw-bold">{total_patients}</h2>
                        <i class="fas fa-users fa-3x opacity-25 position-absolute top-50 end-0 translate-middle-y me-3"></i>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-lg-3">
                <div class="card bg-warning text-dark h-100">
                    <div class="card-body">
                        <h5>Waiting List</h5>
                        <h2 class="display-6 fw-bold">{waiting_patients}</h2>
                        <i class="fas fa-clock fa-3x opacity-25 position-absolute top-50 end-0 translate-middle-y me-3"></i>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-lg-3">
                <div class="card bg-danger text-white h-100">
                    <div class="card-body">
                        <h5>Emergencies</h5>
                        <h2 class="display-6 fw-bold">{emergency_patients}</h2>
                        <i class="fas fa-ambulance fa-3x opacity-25 position-absolute top-50 end-0 translate-middle-y me-3"></i>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-lg-3">
                <div class="card bg-success text-white h-100">
                    <div class="card-body">
                        <h5>Total Doctors</h5>
                        <h2 class="display-6 fw-bold">{len(doctors)}</h2>
                        <i class="fas fa-user-md fa-3x opacity-25 position-absolute top-50 end-0 translate-middle-y me-3"></i>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-5">
            <div class="col-12 col-xl-8">
                <div class="card shadow-sm">
                    <div class="card-header bg-white">
                        <h5 class="mb-0">Recent Patient Activity</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover align-middle">
                                <thead class="table-light">
                                    <tr>
                                        <th>Patient</th>
                                        <th>Department</th>
                                        <th>Status</th>
                                        <th>Time / Schedule</th>
                                    </tr>
                                </thead>
                                <tbody>
    '''
    
    for p in recent_patients:
        if p.status == 'emergency':
            status_color = 'danger'
        elif p.status == 'waiting':
            status_color = 'warning'
        elif p.status == 'in_progress':
            status_color = 'info'
        elif p.status == 'scheduled':
            status_color = 'secondary'
        else:
            status_color = 'success'
        content += f'''
                                    <tr>
                                        <td>
                                            <div class="fw-bold">{p.name}</div>
                                            <small class="text-muted">{p.patient_id}</small>
                                        </td>
                                        <td>{p.department}</td>
                                        <td><span class="badge bg-{status_color}">{p.status.upper()}</span></td>
                                        <td>{format_patient_service_time(p)}</td>
                                    </tr>
        '''
    
    content += f'''
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-12 col-xl-4">
                <div class="card shadow-sm h-100">
                    <div class="card-header bg-white">
                        <h5 class="mb-0">Doctor Directory</h5>
                        <small class="text-muted">View registered doctors and manage their access</small>
                    </div>
                    <div class="card-body">
                        <form method="POST" action="/admin/doctors/add" class="mb-4">
                            <div class="row g-2">
                                <div class="col-12">
                                    <label class="form-label small mb-1">Full Name</label>
                                    <input type="text" class="form-control form-control-sm" name="full_name" placeholder="Dr. Jane Doe" required>
                                </div>
                                <div class="col-12 col-md-6">
                                    <label class="form-label small mb-1">Username</label>
                                    <input type="text" class="form-control form-control-sm" name="username" placeholder="dr_jane" required>
                                </div>
                                <div class="col-12 col-md-6">
                                    <label class="form-label small mb-1">Password</label>
                                    <input type="password" class="form-control form-control-sm" name="password" placeholder="Set password" required>
                                </div>
                                <div class="col-12">
                                    <label class="form-label small mb-1">Department</label>
                                    <select class="form-select form-select-sm" name="department" required>
                                        <option value="" selected disabled>Select department</option>
                                        {department_options}
                                    </select>
                                </div>
                                <div class="col-12 d-grid mt-1">
                                    <button type="submit" class="btn btn-primary btn-sm">
                                        <i class="fas fa-user-plus me-1"></i>Add Doctor
                                    </button>
                                </div>
                            </div>
                        </form>
                        <div class="table-responsive">
                            <table class="table table-sm align-middle mb-0">
                                <thead class="table-light">
                                    <tr>
                                        <th>ID</th>
                                        <th>Name</th>
                                        <th>Username</th>
                                        <th>Department</th>
                                        <th>Status</th>
                                        <th class="text-end">Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {doctor_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    '''
    
    return get_base_html("Admin Dashboard", content, show_chat=False)

@app.route('/admin/doctors/add', methods=['POST'])
@admin_required
def admin_add_doctor():
    """Create a new doctor account from the admin dashboard."""
    full_name = request.form.get('full_name', '').strip()
    username = request.form.get('username', '').strip().lower()
    password = request.form.get('password', '')
    department = request.form.get('department', '').strip()

    if not full_name or not username or not password or not department:
        flash('All doctor fields are required.', 'danger')
        return redirect(url_for('admin_dashboard'))

    if len(password) < 6:
        flash('Doctor password must be at least 6 characters long.', 'danger')
        return redirect(url_for('admin_dashboard'))

    if not Department.query.filter_by(name=department).first():
        flash('Please select a valid department.', 'danger')
        return redirect(url_for('admin_dashboard'))

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        flash('That username already exists. Choose a different username.', 'danger')
        return redirect(url_for('admin_dashboard'))

    doctor = User(
        username=username,
        full_name=full_name,
        department=department,
        role='doctor'
    )
    doctor.set_password(password)
    db.session.add(doctor)
    db.session.commit()

    flash(f'Doctor {html.escape(full_name)} added successfully.', 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/doctors/<int:doctor_id>/delete', methods=['POST'])
@admin_required
def admin_delete_doctor(doctor_id):
    """Remove a doctor account from the system."""
    doctor = db.session.get(User, doctor_id)
    if doctor is None:
        flash('Doctor not found.', 'danger')
        return redirect(url_for('admin_dashboard'))

    if doctor.role != 'doctor':
        flash('Only doctor accounts can be removed from this panel.', 'danger')
        return redirect(url_for('admin_dashboard'))

    doctor_name = doctor.full_name or doctor.username
    db.session.delete(doctor)
    db.session.commit()

    flash(f'Doctor {html.escape(doctor_name)} removed successfully.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/doctor/dashboard')
@doctor_required
def doctor_dashboard():
    """Doctor dashboard with patient management"""
    doctor_name = current_user.full_name
    doctor_dept = current_user.department
    
    # Check if doctor is currently consulting any patient
    current_consultation = Patient.query.filter_by(
        department=doctor_dept,
        status='in_progress',
        doctor_assigned=doctor_name
    ).first()
    
    # Get patients in doctor's department
    patients = Patient.query.filter_by(department=doctor_dept).order_by(
        db.case(
            (Patient.status == 'emergency', 1),
            (Patient.status == 'in_progress', 2),
            (Patient.status == 'waiting', 3),
            (Patient.status == 'scheduled', 4),
            else_=4
        ),
        Patient.consultation_date,
        Patient.check_in_time
    ).all()
    
    content = f'''
    <div class="doctor-dashboard">
        <div class="row align-items-center">
            <div class="col-12 col-md-8 mb-3 mb-md-0">
                <h1 class="display-5 fw-bold mb-3"><i class="fas fa-user-md me-2"></i>Doctor Dashboard</h1>
                <h3>Welcome, {doctor_name}</h3>
                <p class="mb-0">Department: {doctor_dept} | Last Login: {datetime.datetime.now().strftime("%I:%M %p")}</p>
            </div>
            <div class="col-12 col-md-4 text-md-end">
                <a href="/doctor/logout" class="btn btn-light w-100 w-md-auto">
                    <i class="fas fa-sign-out-alt me-2"></i>Logout
                </a>
            </div>
        </div>
    '''
    
    # Show current consultation warning
    if current_consultation:
        content += f'''
        <div class="row mt-3">
            <div class="col-12">
                <div class="alert alert-warning">
                    <div class="d-flex flex-column flex-md-row justify-content-between align-items-center">
                        <div class="text-center text-md-start mb-2 mb-md-0">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            <strong>Currently Consulting:</strong> {current_consultation.name} (Queue: {current_consultation.queue_number})
                            <br>
                            <small>Started at: {current_consultation.check_in_time.strftime("%I:%M %p")}</small>
                        </div>
                        <div>
                            <a href="/doctor/end_consultation/{current_consultation.id}" class="btn btn-danger btn-sm" onclick="return confirmEnd('{current_consultation.name}')">
                                <i class="fas fa-stop-circle me-1"></i> End Consultation
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        '''
    
    if patients:
        content += '''
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Queue</th>
                                    <th>Patient</th>
                                    <th class="d-none d-md-table-cell">Age/Gender</th>
                                    <th class="d-none d-lg-table-cell">Visit Type</th>
                                    <th class="d-none d-sm-table-cell">Time / Schedule</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
        '''
        
        for patient in patients:
            queue_display = patient.queue_number or 'TBD'
            service_time_display = format_patient_service_time(patient)
            
            # Determine status badge color
            if patient.status == 'emergency':
                status_badge = 'danger'
                status_text = 'EMERGENCY'
            elif patient.status == 'in_progress':
                status_badge = 'info'
                status_text = 'IN PROGRESS'
            elif patient.status == 'waiting':
                status_badge = 'warning'
                status_text = 'WAITING'
            elif patient.status == 'scheduled':
                status_badge = 'secondary'
                status_text = 'SCHEDULED'
            else:
                status_badge = 'success'
                status_text = 'COMPLETED'
            
            # Check if this is the patient doctor is currently consulting
            is_current_patient = (patient.status == 'in_progress' and 
                                 patient.doctor_assigned == doctor_name)
            
            content += f'''
                                <tr class="{'current-consultation' if is_current_patient else ''}">
                                    <td><strong>{queue_display}</strong></td>
                                    <td>
                                        {patient.name}
                                        <small class="d-block d-md-none text-muted">{patient.age}/{patient.gender}</small>
                                    </td>
                                    <td class="d-none d-md-table-cell">{patient.age}/{patient.gender}</td>
                                    <td class="d-none d-lg-table-cell">{patient.consultation_type.replace('-', ' ').title()}</td>
                                    <td class="d-none d-sm-table-cell">{service_time_display}</td>
                                    <td>
                                        <span class="badge bg-{status_badge}">{status_text}</span>
                                        {f'<br><small class="d-none d-md-inline">Dr. {patient.doctor_assigned}</small>' if patient.doctor_assigned else ''}
                                    </td>
                                    <td>
                                        <div class="btn-group-vertical btn-group-sm d-flex d-md-inline-flex">
            '''
            
            # Show different buttons based on patient status
            if patient.status == 'waiting':
                if current_consultation:
                    content += f'''
                                            <button class="btn btn-secondary" disabled title="End current consultation first">
                                                <i class="fas fa-play"></i> <span class="d-none d-md-inline">Start</span>
                                            </button>
                    '''
                else:
                    content += f'''
                                            <a href="/doctor/start_consultation/{patient.id}" class="btn btn-primary" title="Start Consultation">
                                                <i class="fas fa-play"></i> <span class="d-none d-md-inline">Start</span>
                                            </a>
                    '''
                content += f'''
                                            <a href="/doctor/declare_emergency/{patient.id}" class="btn btn-danger mt-1 mt-md-0 ms-md-1" title="Declare Emergency" onclick="return confirmEmergency('{patient.name}')">
                                                <i class="fas fa-exclamation-triangle"></i> <span class="d-none d-md-inline">Emergency</span>
                                            </a>
                '''
            elif patient.status == 'in_progress':
                if is_current_patient:
                    content += f'''
                                            <a href="/doctor/end_consultation/{patient.id}" class="btn btn-warning" title="End Consultation" onclick="return confirmEnd('{patient.name}')">
                                                <i class="fas fa-stop-circle"></i> <span class="d-none d-md-inline">End</span>
                                            </a>
                                            <a href="/doctor/complete_consultation/{patient.id}" class="btn btn-success mt-1 mt-md-0 ms-md-1" title="Complete Consultation" onclick="return confirmComplete('{patient.name}')">
                                                <i class="fas fa-check"></i> <span class="d-none d-md-inline">Complete</span>
                                            </a>
                    '''
                else:
                    content += f'''
                                            <button class="btn btn-secondary" disabled title="Being consulted by {patient.doctor_assigned}">
                                                <i class="fas fa-user-md"></i> <span class="d-none d-md-inline">Busy</span>
                                            </button>
                    '''
                content += f'''
                                            <a href="/doctor/declare_emergency/{patient.id}" class="btn btn-danger mt-1 mt-md-0 ms-md-1" title="Declare Emergency" onclick="return confirmEmergency('{patient.name}')">
                                                <i class="fas fa-exclamation-triangle"></i> <span class="d-none d-md-inline">Emergency</span>
                                            </a>
                '''
            elif patient.status == 'emergency':
                if current_consultation and not is_current_patient:
                    content += f'''
                                            <button class="btn btn-secondary" disabled title="End current consultation first">
                                                <i class="fas fa-play"></i> <span class="d-none d-md-inline">Attend</span>
                                            </button>
                    '''
                else:
                    content += f'''
                                            <a href="/doctor/start_consultation/{patient.id}" class="btn btn-primary" title="Attend Emergency">
                                                <i class="fas fa-play"></i> <span class="d-none d-md-inline">Attend</span>
                                            </a>
                    '''
                content += f'''
                                            <a href="/doctor/complete_consultation/{patient.id}" class="btn btn-success mt-1 mt-md-0 ms-md-1" title="Complete Emergency" onclick="return confirmComplete('{patient.name}')">
                                                <i class="fas fa-check"></i> <span class="d-none d-md-inline">Complete</span>
                                            </a>
                '''
            elif patient.status == 'completed':
                content += f'''
                                            <span class="text-muted small">Completed</span>
                '''
            elif patient.status == 'scheduled':
                content += f'''
                                            <span class="text-muted small">Scheduled for {format_consultation_date(patient.consultation_date)}</span>
                '''
            
            content += '''
                                        </div>
                                    </td>
                                </tr>
            '''
        
        content += '''
                            </tbody>
                        </table>
                    </div>
        '''
    else:
        content += '''
                    <div class="text-center py-4">
                        <i class="fas fa-users fa-3x text-muted mb-3"></i>
                        <p class="text-muted">No patients in queue for this department.</p>
                    </div>
        '''
    
    content += '''
                </div>
            </div>
        </div>
    </div>
    
    <!-- Emergency Declaration Form - Responsive -->
    <div class="row mt-4">
        <div class="col-12 col-md-6 mb-4 mb-md-0">
            <div class="card">
                <div class="card-header bg-white">
                    <h5 class="mb-0"><i class="fas fa-exclamation-triangle me-2"></i>Declare New Emergency</h5>
                </div>
                <div class="card-body">
                    <form method="POST" action="/doctor/declare_emergency_manual">
                        <div class="mb-3">
                            <label class="form-label">Patient Name *</label>
                            <input type="text" class="form-control" name="patient_name" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Patient ID (Optional)</label>
                            <input type="text" class="form-control" name="patient_id">
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Department *</label>
                            <select class="form-select" name="department" required>
                                <option value="">Select Department</option>
    '''
    
    # Add department options
    departments = Department.query.all()
    for dept in departments:
        content += f'<option value="{dept.name}" {"selected" if dept.name == doctor_dept else ""}>{dept.name}</option>'
    
    content += '''
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Severity *</label>
                            <select class="form-select" name="severity" required>
                                <option value="critical">Critical</option>
                                <option value="urgent">Urgent</option>
                                <option value="moderate">Moderate</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Reason/Notes *</label>
                            <textarea class="form-control" name="reason" rows="3" required></textarea>
                        </div>
                        <div class="d-grid">
                            <button type="submit" class="btn btn-danger">
                                <i class="fas fa-exclamation-triangle me-2"></i>Declare Emergency
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
        
        <div class="col-12 col-md-6">
            <div class="card">
                <div class="card-header bg-white">
                    <h5 class="mb-0"><i class="fas fa-chart-line me-2"></i>Today's Statistics</h5>
                </div>
                <div class="card-body">
                    <div class="list-group list-group-flush">
    '''
    
    # Get today's statistics
    today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
    
    all_patients_today = Patient.query.filter(
        Patient.check_in_time >= today_start,
        Patient.department == doctor_dept
    ).all()
    
    stats = {
        'total': len(all_patients_today),
        'completed': len([p for p in all_patients_today if p.status == 'completed']),
        'emergency': len([p for p in all_patients_today if p.status == 'emergency']),
        'waiting': len([p for p in all_patients_today if p.status == 'waiting']),
        'in_progress': len([p for p in all_patients_today if p.status == 'in_progress'])
    }
    
    content += f'''
                        <div class="list-group-item d-flex justify-content-between align-items-center">
                            <span>Total Patients Today</span>
                            <span class="badge bg-primary rounded-pill">{stats['total']}</span>
                        </div>
                        <div class="list-group-item d-flex justify-content-between align-items-center">
                            <span>Completed Consultations</span>
                            <span class="badge bg-success rounded-pill">{stats['completed']}</span>
                        </div>
                        <div class="list-group-item d-flex justify-content-between align-items-center">
                            <span>Emergency Cases</span>
                            <span class="badge bg-danger rounded-pill">{stats['emergency']}</span>
                        </div>
                        <div class="list-group-item d-flex justify-content-between align-items-center">
                            <span>Currently In Progress</span>
                            <span class="badge bg-info rounded-pill">{stats['in_progress']}</span>
                        </div>
                        <div class="list-group-item d-flex justify-content-between align-items-center">
                            <span>Waiting Patients</span>
                            <span class="badge bg-warning rounded-pill">{stats['waiting']}</span>
                        </div>
    '''
    
    content += '''
                    </div>
                    <div class="mt-3 alert alert-info small">
                        <i class="fas fa-info-circle me-2"></i>
                        Statistics update automatically every 30 seconds.
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function confirmEmergency(patientName) {
            return window.localizeHospitalConfirm ? window.localizeHospitalConfirm('confirm_emergency', { name: patientName }) : confirm(`Declare EMERGENCY for ${patientName}?`);
        }
        
        function confirmEnd(patientName) {
            return window.localizeHospitalConfirm ? window.localizeHospitalConfirm('confirm_end', { name: patientName }) : confirm(`END consultation with ${patientName}?`);
        }
        
        function confirmComplete(patientName) {
            return window.localizeHospitalConfirm ? window.localizeHospitalConfirm('confirm_complete', { name: patientName }) : confirm(`COMPLETE consultation with ${patientName}?`);
        }
        
        setInterval(() => {
            location.reload();
        }, 30000);
    </script>
    '''
    
    return get_base_html("Doctor Dashboard", content, show_chat=False)

@app.route('/doctor/declare_emergency/<int:patient_id>')
@doctor_required
def declare_emergency(patient_id):
    """Declare emergency for a patient"""
    patient = Patient.query.get_or_404(patient_id)
    
    patient.status = 'emergency'
    patient.department = 'Emergency'
    patient.doctor_assigned = current_user.full_name
    
    emergency = EmergencyLog(
        emergency_id=f"EMG{str(uuid.uuid4())[:8].upper()}",
        patient_id=patient.patient_id,
        patient_name=patient.name,
        department='Emergency',
        declared_by=current_user.full_name,
        severity='critical',
        reason=f"Emergency declared by Dr. {current_user.full_name}"
    )
    
    db.session.add(emergency)
    db.session.commit()
    
    flash(f'Emergency declared for {patient.name}.', 'danger')
    return redirect(url_for('doctor_dashboard'))

@app.route('/doctor/declare_emergency_manual', methods=['POST'])
@doctor_required
def declare_emergency_manual():
    """Manually declare emergency"""
    emergency = EmergencyLog(
        emergency_id=f"EMG{str(uuid.uuid4())[:8].upper()}",
        patient_name=request.form['patient_name'],
        patient_id=request.form.get('patient_id', 'WALK-IN'),
        department=request.form['department'],
        declared_by=current_user.full_name,
        severity=request.form['severity'],
        reason=request.form['reason']
    )
    
    db.session.add(emergency)
    db.session.commit()
    
    flash(f'Emergency declared for {request.form["patient_name"]}.', 'danger')
    return redirect(url_for('doctor_dashboard'))

@app.route('/doctor/start_consultation/<int:patient_id>')
@doctor_required
def start_consultation(patient_id):
    """Start consultation with patient"""
    patient = Patient.query.get_or_404(patient_id)
    doctor_name = current_user.full_name
    
    # Check if doctor is already consulting someone
    current_consultation = Patient.query.filter_by(
        department=patient.department,
        status='in_progress',
        doctor_assigned=doctor_name
    ).first()
    
    if current_consultation:
        flash(f'You are currently consulting {current_consultation.name}.', 'warning')
        return redirect(url_for('doctor_dashboard'))
    
    if patient.status in ['waiting', 'emergency']:
        patient.status = 'in_progress'
        patient.doctor_assigned = doctor_name
        db.session.commit()
        flash(f'Started consultation with {patient.name}.', 'success')
    else:
        flash(f'Cannot start consultation. Patient status: {patient.status}.', 'warning')
    
    return redirect(url_for('doctor_dashboard'))

@app.route('/doctor/end_consultation/<int:patient_id>')
@doctor_required
def end_consultation(patient_id):
    """End current consultation"""
    patient = Patient.query.get_or_404(patient_id)
    doctor_name = current_user.full_name
    
    if patient.status == 'in_progress' and patient.doctor_assigned == doctor_name:
        patient.status = 'waiting'
        patient.doctor_assigned = None
        db.session.commit()
        flash(f'Ended consultation with {patient.name}.', 'warning')
    else:
        flash(f'Cannot end consultation.', 'danger')
    
    return redirect(url_for('doctor_dashboard'))

@app.route('/doctor/complete_consultation/<int:patient_id>')
@doctor_required
def complete_consultation(patient_id):
    """Mark patient consultation as completed"""
    patient = Patient.query.get_or_404(patient_id)
    
    if patient.status in ['in_progress', 'emergency']:
        patient.status = 'completed'
        patient.completion_time = datetime.datetime.utcnow()
        db.session.commit()
        flash(f'Completed consultation with {patient.name}.', 'success')
    else:
        flash(f'Cannot complete consultation.', 'warning')
    
    return redirect(url_for('doctor_dashboard'))

@app.route('/doctor/logout')
@login_required
def doctor_logout():
    """Logout doctor"""
    flash('Logging out...', 'info')
    print(f"[DEBUG LOG] Logout route hit. User: {current_user.username if current_user.is_authenticated else 'Anonymous'}")
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Patient registration"""
    # Get department from query parameter if provided
    selected_dept = request.args.get('department', '')
    
    if request.method == 'POST':
        print(f"\n[DEBUG] Registration POST request received!")
        print(f"  Form Keys: {list(request.form.keys())}")
        print(f"  Captured Face ID: {request.form.get('captured_face_id')}")
        print(f"  Face Descriptor (first 50 chars): {str(request.form.get('face_descriptor'))[:50]}")
        try:
            consultation_date = parse_consultation_date(request.form['consultation_date'])
            today = datetime.date.today()
            if consultation_date < today:
                raise ValueError("Consultation date cannot be in the past.")

            is_future_consultation = consultation_date > today
            patient_status = 'scheduled' if is_future_consultation else 'waiting'
            consultation_time, queue_position = estimate_consultation_slot(
                request.form['department'],
                consultation_date
            )
            queue_number = (
                None
                if is_future_consultation
                else f"{request.form['department'][:3].upper()}{Patient.query.count() + 1:03d}"
            )
            check_in_time = None if is_future_consultation else datetime.datetime.utcnow()

            # Get face data from Temporary table or form
            face_descriptor = None
            face_image = None
            
            captured_face_id = request.form.get('captured_face_id')
            if captured_face_id:
                try:
                    captured_face = CapturedFace.query.get(int(captured_face_id))
                    if captured_face:
                        face_descriptor = captured_face.face_descriptor
                        face_image = captured_face.face_image
                        # Optionally delete the temp record later or keep as a log
                except Exception as e:
                    print(f"[!] Error fetching CapturedFace: {e}")
            else:
                # Fallback to direct inputs if available
                face_descriptor = request.form.get('face_descriptor')
                face_image = request.form.get('face_image')

            existing_patient_id = request.form.get('existing_patient_id', '').strip()
            existing_patient, identity_source = _find_existing_patient_for_registration(
                existing_patient_id=existing_patient_id,
                aadhaar_number=request.form.get('aadhaar_number', ''),
                phone=request.form.get('phone', ''),
                face_descriptor=face_descriptor,
                name=request.form.get('name', '')
            )

            returning_patient = existing_patient is not None
            if returning_patient:
                patient = existing_patient
                patient_id = patient.patient_id
                patient.name = request.form['name']
                patient.age = int(request.form['age'])
                patient.gender = request.form['gender']
                patient.phone = request.form['phone']
                patient.email = request.form.get('email', '')
                patient.aadhaar_number = request.form.get('aadhaar_number', '')
                patient.department = request.form['department']
                patient.consultation_type = request.form['consultation_type']
                patient.consultation_date = consultation_date
                patient.consultation_time = consultation_time
                patient.queue_number = queue_number
                patient.check_in_time = check_in_time
                patient.status = patient_status
                patient.doctor_assigned = None
                patient.completion_time = None
                if face_descriptor is not None:
                    patient.face_descriptor = face_descriptor
                if face_image is not None:
                    patient.face_image = face_image
            else:
                # Generate a new patient ID only for a brand-new patient
                patient_id = f"PAT{str(uuid.uuid4())[:8].upper()}"
                patient = Patient(
                    patient_id=patient_id,
                    name=request.form['name'],
                    age=int(request.form['age']),
                    gender=request.form['gender'],
                    phone=request.form['phone'],
                    email=request.form.get('email', ''),
                    aadhaar_number=request.form.get('aadhaar_number', ''),
                    department=request.form['department'],
                    consultation_type=request.form['consultation_type'],
                    consultation_date=consultation_date,
                    consultation_time=consultation_time,
                    queue_number=queue_number,
                    check_in_time=check_in_time,
                    status=patient_status,
                    face_descriptor=face_descriptor,
                    face_image=face_image
                )
                db.session.add(patient)

            print(f"[DEBUG] Aadhaar number received: '{request.form.get('aadhaar_number', 'NOT RECEIVED')}'")
            print(f"[DEBUG] Returning patient: {returning_patient}  source={identity_source or 'new'}  patient_id={patient_id}")

            user = User.query.get(patient.user_id) if patient.user_id else None
            if user is None:
                user = User(username=patient_id, role='patient', full_name=patient.name, department=patient.department)
                user.set_password(patient_id)
                db.session.add(user)
                db.session.flush()
                patient.user_id = user.id
            else:
                user.username = patient_id
                user.role = 'patient'
                user.full_name = patient.name
                user.department = patient.department

            db.session.commit()

            consultation_date_text = format_consultation_date(patient.consultation_date)
            consultation_time_text = format_consultation_time(patient.consultation_time)
            queue_label = "Appointment Status" if is_future_consultation else "Queue Number"
            queue_value = "Scheduled" if is_future_consultation else (patient.queue_number or "Pending")
            timing_line = (
                f"<p><strong>Consultation Date:</strong> {consultation_date_text}</p>"
                f"<p><strong>Estimated Consultation Time:</strong> {consultation_time_text}</p>"
                f"<p><strong>Estimated Queue Position:</strong> {queue_position}</p>"
                f"<p><strong>Status:</strong> {patient.status.replace('_', ' ').title()}</p>"
                f"<p><strong>Queue Number:</strong> Will be assigned on the consultation day.</p>"
                if is_future_consultation else
                f"<p><strong>Consultation Date:</strong> {consultation_date_text}</p>"
                f"<p><strong>Estimated Consultation Time:</strong> {consultation_time_text}</p>"
                f"<p><strong>Queue Position:</strong> {queue_position}</p>"
                f"<p><strong>Check-in Time:</strong> {patient.check_in_time.strftime('%I:%M %p')}</p>"
            )
            secondary_action = (
                '<a href="/queue" class="btn btn-outline-primary">Check Queue Status</a>'
                if not is_future_consultation else
                '<a href="/navigation" class="btn btn-outline-primary">View Hospital Navigation</a>'
            )
            sms_result = {'sent': False, 'reason': 'not_attempted'}
            try:
                sms_result = send_registration_sms(patient)
                print(f"[SMS] Registration SMS result for {patient.patient_id}: {sms_result}")
            except Exception as sms_error:
                sms_result = {'sent': False, 'reason': str(sms_error)}
                print(f"[SMS] Failed to send registration SMS for {patient.patient_id}: {sms_error}")

            sms_reason = sms_result.get('reason', 'unknown')
            sms_alert = (
                f'<div class="alert alert-success mt-3 mb-0"><i class="fas fa-sms me-2"></i>'
                f'Confirmation SMS sent to {patient.phone}.</div>'
                if sms_result.get('sent') else
                f'<div class="alert alert-warning mt-3 mb-0"><i class="fas fa-exclamation-circle me-2"></i>'
                f'Registration saved, but SMS confirmation could not be sent right now.'
                f'<br><small class="text-muted">Reason: {sms_reason}</small></div>'
            )
            
            # Success page
            success_content = f'''
            <div class="row justify-content-center py-3 py-md-5">
                <div class="col-12 col-lg-8">
                    <div class="alert alert-success">
                        <h4 class="alert-heading"><i class="fas fa-check-circle me-2"></i>Registration Successful!</h4>
                        <hr>
                        <div class="row">
                            <div class="col-12 col-md-6 mb-3 mb-md-0">
                                <p><strong>Patient ID:</strong></p>
                                <h3 class="text-primary text-break">{patient_id}</h3>
                            </div>
                            <div class="col-12 col-md-6">
                                <p><strong>{queue_label}:</strong></p>
                                <h3 class="text-primary">{queue_value}</h3>
                            </div>
                        </div>
                        <div class="mt-3">
                            <p><strong>Department:</strong> {patient.department}</p>
                            <p><strong>Aadhaar Number:</strong> {patient.aadhaar_number if patient.aadhaar_number else 'Not Provided'}</p>
                            {timing_line}
                        </div>
                        <div class="mt-4 d-grid d-md-block">
                            <a href="/" class="btn btn-primary me-md-2 mb-2 mb-md-0">Return to Home</a>
                            {secondary_action}
                        </div>
                        {sms_alert}
                    </div>
                </div>
            </div>
            '''
            
            return get_base_html("Registration Successful", success_content, show_chat=True)
            
        except Exception as e:
            error_content = f'''
            <div class="row justify-content-center">
                <div class="col-12 col-lg-8">
                    <div class="alert alert-danger">
                        <h4><i class="fas fa-exclamation-triangle me-2"></i>Registration Failed</h4>
                        <p>Error: {str(e)}</p>
                        <a href="/register" class="btn btn-warning mt-2">Try Again</a>
                    </div>
                </div>
            </div>
            '''
            return get_base_html("Registration Error", error_content, show_chat=True)

    # GET request - show registration form
    departments = Department.query.all()
    today_iso = datetime.date.today().isoformat()

    dept_options = ""
    for dept in departments:
        selected = 'selected' if dept.name == selected_dept else ''
        dept_options += f'<option value="{dept.name}" {selected}>{dept.name}</option>'

    content = f'''
    <div class="row justify-content-center">
        <div class="col-12 col-lg-8">
            <div class="mb-4">
                <h2 class="text-center mb-4"><i class="fas fa-user-plus me-2"></i>Patient Registration</h2>
                <p class="text-center text-muted">Complete both steps below to register and choose your consultation date</p>
            </div>

            <!-- STEP 1: Face Capture -->
            <div class="card mb-4 border-primary">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0"><i class="fas fa-camera me-2"></i>Step 1: Face Capture <span class="badge bg-warning text-dark ms-2" id="step1Badge">Required</span></h5>
                </div>
                <div class="card-body p-3 p-md-4">
                    <div class="alert alert-warning small mb-3">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Face capture is mandatory. Please capture your face first before filling the registration form.
                    </div>
                    <div class="row">
                        <div class="col-12 col-md-6">
                            <div class="card bg-light">
                                <div class="card-body text-center p-2">
                                    <div style="position: relative; width: 100%; display: inline-block;">
                                        <video id="faceVideo" width="100%" height="220" autoplay muted style="border: 2px solid #dee2e6; border-radius: 8px; display: block; background:#000;"></video>
                                        <canvas id="faceCircleOverlay" width="300" height="220" style="position: absolute; top: 0; left: 0; width: 100%; height: 220px; border-radius: 8px; cursor: pointer;"></canvas>
                                    </div>
                                    <div id="faceMessage" class="mt-2"></div>
                                </div>
                            </div>
                        </div>
                        <div class="col-12 col-md-6 d-flex flex-column justify-content-center mt-3 mt-md-0">
                            <p class="text-muted small mb-3"><i class="fas fa-info-circle me-1"></i>Position your face clearly in the camera circle, then click <strong>Capture Face</strong>.</p>
                            <div class="d-grid gap-2">
                                <button type="button" class="btn btn-outline-primary btn-lg" id="startCameraBtn" onclick="startFaceCamera()">
                                    <i class="fas fa-video me-2"></i>Start Camera
                                </button>
                                <button type="button" class="btn btn-success btn-lg" id="captureFaceBtn" onclick="captureFace()" disabled>
                                    <i class="fas fa-camera me-2"></i>Capture Face
                                </button>
                                <button type="button" class="btn btn-outline-secondary" id="retryFaceBtn" onclick="retryFace()" style="display: none;">
                                    <i class="fas fa-redo me-2"></i>Retry
                                </button>
                            </div>
                            <div id="faceStatus" class="mt-3" style="display: none;"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- STEP 2: Registration Form -->
            <div class="card" id="step2Card" style="display:none; opacity:1; pointer-events:auto;">
                <div class="card-header bg-success text-white" id="step2Header">
                    <h5 class="mb-0" id="step2HeaderText"><i class="fas fa-file-alt me-2"></i>Step 2: Patient Details</h5>
                </div>
                <div class="card-body p-3 p-md-4">
                    <form method="POST" id="registrationForm">
                        <!-- Hidden inputs for face data (must be inside form) -->
                        <input type="hidden" name="captured_face_id" id="capturedFaceIdInput">
                        <input type="hidden" name="face_descriptor" id="faceDescriptorInput">
                        <input type="hidden" name="face_image" id="faceImageInput">
                        <input type="hidden" name="existing_patient_id" id="existingPatientIdInput">
                        <div class="row g-3" id="personalDetailsSection">
                            <div class="col-12">
                                <label class="form-label">Full Name *</label>
                                <input type="text" class="form-control" name="name" required placeholder="Enter your full name">
                            </div>

                            <div class="col-6 col-md-3">
                                <label class="form-label">Age *</label>
                                <input type="number" class="form-control" name="age" required min="0" max="150">
                            </div>

                            <div class="col-6 col-md-3">
                                <label class="form-label">Gender *</label>
                                <select class="form-select" name="gender" required>
                                    <option value="">Select</option>
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                    <option value="Other">Other</option>
                                </select>
                            </div>

                            <div class="col-12 col-md-6">
                                <label class="form-label">Phone *</label>
                                <input type="tel" class="form-control" name="phone" required placeholder="10-digit mobile number">
                            </div>

                            <div class="col-12 col-md-6">
                                <label class="form-label">Email (Optional)</label>
                                <input type="email" class="form-control" name="email" placeholder="your@email.com">
                            </div>

                            <div class="col-12 col-md-6">
                                <label class="form-label d-flex justify-content-between">
                                    <span>Aadhaar Number (Optional)</span>
                                    <span id="aadhaarCounter" class="badge bg-secondary">0/12</span>
                                </label>
                                <input type="text" class="form-control" name="aadhaar_number" id="aadhaarInput" placeholder="12-digit Aadhaar number" pattern="[0-9]{{12}}" maxlength="12" title="Please enter a valid 12-digit Aadhaar number" oninput="updateAadhaarCounter(this)">
                                <div class="form-text">Exactly 12 digits (no spaces or hyphens)</div>
                            </div>
                        </div> <!-- End of personalDetailsSection -->

                        <div class="row g-3 mt-1">
                            <div class="col-12 col-md-4">
                                <label class="form-label">Department *</label>
                                <select class="form-select" name="department" required id="departmentSelect">
                                    <option value="">Select Department</option>
                                    {dept_options}
                                </select>
                            </div>

                            <div class="col-12 col-md-4">
                                <label class="form-label">Visit Type *</label>
                                <select class="form-select" name="consultation_type" required>
                                    <option value="">Select Type</option>
                                    <option value="new">New Consultation</option>
                                    <option value="follow-up">Follow-up Visit</option>
                                </select>
                            </div>

                            <div class="col-12 col-md-4">
                                <label class="form-label">Consultation Date *</label>
                                <input type="date" class="form-control" name="consultation_date" id="consultationDateInput" required min="{today_iso}" value="{today_iso}">
                                <div class="form-text">Choose today for immediate check-in or a future date to schedule the visit. Estimated consultation time will be assigned automatically from the queue.</div>
                            </div>
                        </div>

                        <div class="mt-4 alert alert-info small" id="deptInfo" style="display: none;">
                            <i class="fas fa-info-circle me-2"></i>
                            <span id="deptInfoText"></span>
                        </div>

                        <div class="d-grid mt-4">
                            <button type="submit" class="btn btn-primary btn-lg" id="registerBtn">
                                <i class="fas fa-user-plus me-2"></i><span id="submitBtnText">Complete Registration</span>
                            </button>
                        </div>
                    </form>
                </div>
            </div>
            
            <!-- Quick Info Cards -->
            <div class="row mt-4 g-3">
                <div class="col-6 col-md-3">
                    <div class="card text-center p-2">
                        <i class="fas fa-clock text-primary mb-2"></i>
                        <h6 class="mb-0">Avg. Wait Time</h6>
                        <small>15-20 mins</small>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="card text-center p-2">
                        <i class="fas fa-file-invoice text-success mb-2"></i>
                        <h6 class="mb-0">Fee</h6>
                        <small>₹500</small>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="card text-center p-2">
                        <i class="fas fa-id-card text-warning mb-2"></i>
                        <h6 class="mb-0">Documents</h6>
                        <small>ID Proof</small>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="card text-center p-2">
                        <i class="fas fa-ambulance text-danger mb-2"></i>
                        <h6 class="mb-0">Emergency</h6>
                        <small><a href="tel:112" class="text-danger text-decoration-none fw-semibold">Call 112</a></small>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Show department info when selected
        document.getElementById('departmentSelect').addEventListener('change', function() {{
            const deptInfo = document.getElementById('deptInfo');
            const deptInfoText = document.getElementById('deptInfoText');
            
            if (this.value) {{
                const departments = {json.dumps([{'name': d.name, 'location': d.location, 'description': d.description} for d in departments])};
                const dept = departments.find(d => d.name === this.value);
                if (dept) {{
                    deptInfoText.innerHTML = `<strong>${{dept.name}}:</strong> ${{dept.description}}<br>📍 ${{dept.location}}`;
                    deptInfo.style.display = 'block';
                }}
            }} else {{
                deptInfo.style.display = 'none';
            }}
        }});
        
        // Aadhaar counter update
        function updateAadhaarCounter(input) {{
            const counter = document.getElementById('aadhaarCounter');
            const val = input.value.replace(/\\D/g, '');
            input.value = val; // Force numeric only
            const len = val.length;
            
            counter.innerText = `${{len}}/12`;
            if (len === 12) {{
                counter.className = 'badge bg-success';
                input.classList.remove('is-invalid');
                input.classList.add('is-valid');
            }} else if (len > 0) {{
                counter.className = 'badge bg-warning text-dark';
                input.classList.remove('is-valid');
            }} else {{
                counter.className = 'badge bg-secondary';
                input.classList.remove('is-valid', 'is-invalid');
            }}
        }}

        // Form validation
        document.getElementById('registrationForm').addEventListener('submit', function(e) {{
            const phone = document.querySelector('input[name="phone"]').value;
            if (!/^\\d{{10}}$/.test(phone)) {{
                e.preventDefault();
                if (window.localizeHospitalAlert) {{ window.localizeHospitalAlert('invalid_phone'); }} else {{ alert('Please enter a valid 10-digit phone number'); }}
                return;
            }}
            
            // Validate Aadhaar if provided
            const aadhaar = document.getElementById('aadhaarInput').value;
            if (aadhaar && aadhaar.length !== 12) {{
                e.preventDefault();
                if (window.localizeHospitalAlert) {{ window.localizeHospitalAlert('invalid_aadhaar'); }} else {{ alert('Aadhaar number must be exactly 12 digits'); }}
                document.getElementById('aadhaarInput').focus();
                return;
            }}

            const consultationDateInput = document.getElementById('consultationDateInput');
            if (!consultationDateInput.value || consultationDateInput.value < consultationDateInput.min) {{
                e.preventDefault();
                if (window.localizeHospitalAlert) {{ window.localizeHospitalAlert('invalid_date'); }} else {{ alert('Please choose a valid consultation date.'); }}
                consultationDateInput.focus();
                return;
            }}
            
            // Check if face is captured
            const faceDescriptor = document.getElementById('faceDescriptorInput').value;
            if (!faceDescriptor) {{
                e.preventDefault();
                if (window.localizeHospitalAlert) {{ window.localizeHospitalAlert('face_missing'); }} else {{ alert('Please capture your face before registering'); }}
                return;
            }}
        }});
        
        // Face Detection Variables
        let faceCaptured = false;
        let animationFrameId = null;
        
        // Display messages to user
        async function handleMessage(message, type = 'info') {{
            const faceMessage = document.getElementById('faceMessage');
            if (faceMessage) {{
                faceMessage.className = `alert alert-${{type}} small mt-2 notranslate`;
                faceMessage.setAttribute('translate', 'no');
                faceMessage.dataset.originalMessage = message;
                const targetLanguage = window.getHospitalLanguage ? window.getHospitalLanguage() : 'en';
                faceMessage.innerHTML = window.translateHospitalTextAsync
                    ? await window.translateHospitalTextAsync(message, targetLanguage)
                    : message;
                faceMessage.style.display = 'block';
            }} else {{
                console.log("Face Message: " + message);
            }}
        }}
        
        // Draw circle guide on canvas overlay
        function drawFaceCircleGuide() {{
            const canvas = document.getElementById('faceCircleOverlay');
            const video = document.getElementById('faceVideo');
            
            if (!canvas || !video) return;
            
            const ctx = canvas.getContext('2d');
            
            // Set canvas size to match video dimensions displayed
            canvas.width = video.offsetWidth;
            canvas.height = video.offsetHeight;
            
            // Clear previous drawing
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Calculate circle dimensions (80% of smallest dimension)
            const diameter = Math.min(canvas.width, canvas.height) * 0.75;
            const radius = diameter / 2;
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            // Draw outer circle guide (semi-transparent)
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
            ctx.strokeStyle = 'rgba(76, 175, 80, 0.7)';
            ctx.lineWidth = 3;
            ctx.stroke();
            
            // Draw inner dashed circle
            ctx.beginPath();
            ctx.setLineDash([5, 5]);
            ctx.arc(centerX, centerY, radius - 10, 0, 2 * Math.PI);
            ctx.strokeStyle = 'rgba(76, 175, 80, 0.5)';
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Draw center point
            ctx.beginPath();
            ctx.arc(centerX, centerY, 4, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(76, 175, 80, 0.8)';
            ctx.fill();
            
            // Draw instruction text
            ctx.fillStyle = 'rgba(76, 175, 80, 0.9)';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Position your face', centerX, centerY - radius - 25);
            ctx.fillText('inside the circle', centerX, centerY - radius - 5);
            
            // Continue animation
            animationFrameId = requestAnimationFrame(drawFaceCircleGuide);
        }}
        
        // Stop drawing circle guide
        function stopDrawingCircleGuide() {{
            if (animationFrameId) {{
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }}
            const canvas = document.getElementById('faceCircleOverlay');
            if (canvas) {{
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }}
        }}
        
        // Start camera for face detection
        async function startFaceCamera() {{
            try {{
                handleMessage('Requesting camera access...', 'info');
                
                const stream = await navigator.mediaDevices.getUserMedia({{ 
                    video: {{ 
                        width: 640, 
                        height: 480,
                        facingMode: 'user' 
                    }} 
                }});
                
                const faceVideo = document.getElementById('faceVideo');
                faceVideo.srcObject = stream;
                
                faceVideo.onloadedmetadata = () => {{
                    handleMessage('[OK] Camera ready! Position your face inside the circle and click \"Capture Face\".', 'success');
                    document.getElementById('startCameraBtn').disabled = true;
                    document.getElementById('captureFaceBtn').disabled = false;
                    drawFaceCircleGuide();
                }};
                
            }} catch (err) {{
                console.error('Camera error:', err);
                if (err.name === 'NotAllowedError') {{
                    handleMessage('[X] Camera access denied. Please allow camera access and try again.', 'danger');
                }} else if (err.name === 'NotFoundError') {{
                    handleMessage('[X] No camera found on this device.', 'danger');
                }} else {{
                    handleMessage('[X] Camera error: ' + err.message, 'danger');
                }}
            }}
        }}
        
        // Capture face from video
        async function captureFace() {{
            try {{
                const faceVideo = document.getElementById('faceVideo');
                const canvas = document.createElement('canvas');
                
                canvas.width = faceVideo.videoWidth;
                canvas.height = faceVideo.videoHeight;
                
                const ctx = canvas.getContext('2d');
                ctx.drawImage(faceVideo, 0, 0);
                
                const imageData = canvas.toDataURL('image/jpeg', 0.9);
                
                handleMessage('Detecting face...', 'info');
                document.getElementById('captureFaceBtn').disabled = true;
                const existingPatientInput = document.getElementById('existingPatientIdInput');
                if (existingPatientInput) {{
                    existingPatientInput.value = '';
                }}
                
                // Send image to backend for face detection
                const response = await fetch('/api/register/detect-face', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{image: imageData}})
                }});
                
                const result = await response.json();
                
                if (!response.ok) {{
                    handleMessage('[X] ' + (result.error || 'Face detection failed'), 'danger');
                    document.getElementById('captureFaceBtn').disabled = false;
                    return;
                }}
                
                // Success - store face data
                if (result.captured_face_id) {{
                    const capturedFaceInput = document.getElementById('capturedFaceIdInput');
                    console.log("Captured Face ID from server: " + result.captured_face_id);
                    if(capturedFaceInput) {{
                        capturedFaceInput.value = result.captured_face_id;
                    }}
                }}
                document.getElementById('faceDescriptorInput').value = JSON.stringify(result.embedding);
                document.getElementById('faceImageInput').value = imageData;
                faceCaptured = true;
                
                // Stop camera and circle guide
                stopDrawingCircleGuide();
                faceVideo.srcObject.getTracks().forEach(track => track.stop());
                
                // Update UI
                document.getElementById('captureFaceBtn').style.display = 'none';
                document.getElementById('retryFaceBtn').style.display = 'inline-block';
                document.getElementById('registerBtn').disabled = false;

                // Unlock and show Step 2 form card
                const step2Card = document.getElementById('step2Card');
                if (step2Card) {{
                    step2Card.style.display = 'block';
                }}
                
                // Show success status
                const faceStatus = document.getElementById('faceStatus');
                faceStatus.style.display = 'block';
                faceStatus.innerHTML = '<i class="fas fa-check-circle me-2"></i>Face captured successfully! Processing...';
                faceStatus.className = 'alert alert-success small';
                
                handleMessage('[OK] Face captured successfully! Processing...', 'success');
                
                // Check for existing patient
                try {{
                    const matchResponse = await fetch('/api/register/match-face', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{embedding: result.embedding}})
                    }});
                    
                    const matchData = await matchResponse.json();
                    const personalSection = document.getElementById('personalDetailsSection');
                    const headerText = document.getElementById('step2HeaderText');
                    const submitText = document.getElementById('submitBtnText');
                    
                    if (matchData.found) {{
                        const patient = matchData.patient;
                        if (existingPatientInput) {{
                            existingPatientInput.value = patient.id;
                        }}
                        faceStatus.innerHTML = `<i class="fas fa-user-check me-2"></i>Welcome back, <strong>${{patient.name}}</strong>! Profile loaded.`;
                        faceStatus.className = 'alert alert-info small';
                        
                        // Hide personal details for existing patients to skip filling them out again
                        if(personalSection) personalSection.style.display = 'none';
                        if(headerText) {{
                            headerText.innerHTML = `<i class="fas fa-door-open me-2"></i>Welcome back, ${{patient.name}}!`;
                            document.getElementById('step2Header').className = 'card-header bg-info text-white';
                        }}
                        if(submitText) submitText.textContent = "Confirm Visit & Check In";
                        
                        // Set Visit Type to follow-up automatically for returning patients
                        const visitTypeSelect = document.querySelector('select[name="consultation_type"]');
                        if (visitTypeSelect) visitTypeSelect.value = 'follow-up';
                        
                        // Auto-fill hidden form fields
                        document.querySelector('input[name="name"]').value = patient.name;
                        document.querySelector('input[name="age"]').value = patient.age;
                        document.querySelector('select[name="gender"]').value = patient.gender;
                        document.querySelector('input[name="phone"]').value = patient.phone;
                        document.querySelector('input[name="email"]').value = patient.email || '';
                        
                        handleMessage(`Welcome back ${{patient.name}}! Please select the department and consultation date for this visit.`, 'success');
                    }} else {{
                        if (existingPatientInput) {{
                            existingPatientInput.value = '';
                        }}
                        faceStatus.innerHTML = `<i class="fas fa-user-plus me-2"></i>New profile. Please complete the registration.`;
                        if(personalSection) personalSection.style.display = 'flex';
                        if(headerText) headerText.innerHTML = `<i class="fas fa-file-alt me-2"></i>Step 2: New Patient Details`;
                        if(submitText) submitText.textContent = "Complete Registration";
                        handleMessage('New patient detected. Please fill in your details and choose a consultation date.', 'info');
                    }}
                    
                    // Always scroll to step 2 afterward
                    if (step2Card) step2Card.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                }} catch (matchErr) {{
                    console.error('Match error:', matchErr);
                    const existingPatientInput = document.getElementById('existingPatientIdInput');
                    if (existingPatientInput) {{
                        existingPatientInput.value = '';
                    }}
                    const personalSection = document.getElementById('personalDetailsSection');
                    if(personalSection) personalSection.style.display = 'flex';
                    if (step2Card) step2Card.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                    handleMessage('Face stored successfully! Please fill in your details.', 'success');
                }}
                
            }} catch (err) {{
                console.error('Face capture error:', err);
                handleMessage('[X] Error: ' + (err.message || err), 'danger');
                document.getElementById('captureFaceBtn').disabled = false;
            }}
        }}
        
        // Retry face capture
        function retryFace() {{
            // Reset state
            faceCaptured = false;
            document.getElementById('faceDescriptorInput').value = '';
            document.getElementById('faceImageInput').value = '';
            const existingPatientInput = document.getElementById('existingPatientIdInput');
            if (existingPatientInput) {{
                existingPatientInput.value = '';
            }}
            
            // Reset UI
            document.getElementById('faceStatus').style.display = 'none';
            document.getElementById('startCameraBtn').disabled = false;
            document.getElementById('captureFaceBtn').disabled = true;
            document.getElementById('captureFaceBtn').style.display = 'inline-block';
            document.getElementById('retryFaceBtn').style.display = 'none';

            // Hide Step 2 card
            const step2Card = document.getElementById('step2Card');
            if (step2Card) {{
                step2Card.style.display = 'none';
            }}
            
            const step1Badge = document.getElementById('step1Badge');
            if (step1Badge) {{
                step1Badge.textContent = 'Required';
                step1Badge.className = 'badge bg-warning text-dark ms-2';
            }}
            
            handleMessage('', 'info');
        }}

        // Auto-start camera when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            startFaceCamera();
        }});
    </script>
    '''
    
    return get_base_html("Patient Registration", content, show_chat=True)

@app.route('/api/register/detect-face', methods=['POST'])
def detect_face():
    """
    Detect a face in the submitted webcam image and return its 512-D FaceNet
    embedding.  Tries backends in order:
      1. facenet-pytorch  (MTCNN → InceptionResNetV1 VGGFace2)
      2. DeepFace Facenet512 + MTCNN detector
      3. DeepFace Facenet512 + OpenCV detector (relaxed)
      4. OpenCV Haar + pixel descriptor (512-D, last resort)
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        raw_b64 = data['image']
        if ',' in raw_b64:
            raw_b64 = raw_b64.split(',', 1)[1]
        image_bytes = base64.b64decode(raw_b64)

        embedding, model_name = _extract_embedding_from_image_bytes(image_bytes)

        if embedding is None:
            print("[FACE DETECT] All backends failed – no face extracted")
            return jsonify({
                'error': 'No face detected. Please face the camera squarely in good lighting and retry.'
            }), 400

        print(f"[FACE DETECT] OK  dim={len(embedding)}  model={model_name}")

        captured_record = CapturedFace(
            face_descriptor=json.dumps(embedding),
            face_image=raw_b64
        )
        db.session.add(captured_record)
        db.session.commit()

        return jsonify({
            'success': True,
            'embedding': embedding,
            'captured_face_id': captured_record.id,
            'model': model_name,
            'dimensions': len(embedding)
        })

    except Exception as e:
        print(f"[FACE DETECT ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/register/match-face', methods=['POST'])
def match_face():
    """
    1:N face identification against all registered patients.

    Accepts a 512-D (or 128-D fallback) embedding and compares it via
    cosine similarity against every stored patient descriptor.
    Returns the best match if it clears the trained confidence gates, plus
    full patient details so the kiosk can pre-fill the check-in form.
    """
    try:
        data = request.get_json()
        if not data or 'embedding' not in data:
            return jsonify({'error': 'No embedding provided'}), 400

        probe = np.array(data['embedding'], dtype=np.float32)
        probe_norm = np.linalg.norm(probe)
        if probe_norm < 1e-6:
            return jsonify({'error': 'Invalid (zero-norm) embedding received'}), 400
        probe_unit = probe / probe_norm
        probe_dim  = len(probe)

        # ── Load trained thresholds (use safe defaults if model not yet trained) ─
        cfg          = get_face_model_config()
        COSINE_TH    = cfg['cosine_threshold']
        EUCLIDEAN_TH = cfg['euclidean_threshold']
        GAP_TH       = cfg['similarity_gap']

        # ── 1:N exhaustive search ──────────────────────────────────────────
        patients   = Patient.query.filter(Patient.face_descriptor.isnot(None)).all()
        all_scores = []   # list of (cosine_sim, euclidean_dist, patient)

        for patient in patients:
            try:
                stored = np.array(json.loads(patient.face_descriptor), dtype=np.float32)
                if len(stored) != probe_dim:   # skip incompatible embeddings
                    continue
                sn = np.linalg.norm(stored)
                if sn < 1e-6:
                    continue
                su = stored / sn

                cos  = float(np.dot(probe_unit, su))
                dist = float(np.linalg.norm(probe_unit - su))  # euclidean on unit vectors
                all_scores.append((cos, dist, patient))

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                continue

        all_scores.sort(key=lambda x: x[0], reverse=True)   # best cosine first

        print(f"\n[FACE MATCH] probe_dim={probe_dim}  candidates={len(all_scores)}")
        for cos, dist, p in all_scores[:5]:
            print(f"  {p.name:30s}  cosine={cos:.4f}  euclidean={dist:.4f}")

        if not all_scores:
            return jsonify({'found': False,
                            'message': 'No registered patients found in database.'})

        best_cos, best_dist, best_patient = all_scores[0]
        second_cos = all_scores[1][0] if len(all_scores) > 1 else -1.0
        gap        = best_cos - second_cos

        # ── Decision ───────────────────────────────────────────────────────
        excellent  = best_cos >= FACE_MATCH_EXCELLENT_THRESHOLD   # Tier 1
        std_cosine = best_cos >= COSINE_TH                        # Tier 2a
        std_dist   = best_dist <= EUCLIDEAN_TH                    # Tier 2b
        confirmed  = excellent or (std_cosine and std_dist)

        print(f"  Thresholds  cosine>={COSINE_TH}  euclidean<={EUCLIDEAN_TH}  gap>={GAP_TH}")
        print(f"  excellent={excellent}  cosine={std_cosine}  dist={std_dist}")
        print(f"  RESULT: {'MATCH [OK]' if confirmed else 'NO MATCH [X]'}  ({best_patient.name})")

        if confirmed:
            p = best_patient
            return jsonify({
                'found': True,
                'patient': {
                    'id':                p.patient_id,
                    'name':              p.name,
                    'age':               p.age,
                    'gender':            p.gender,
                    'phone':             p.phone,
                    'email':             p.email or '',
                    'aadhaar_number':    p.aadhaar_number or '',
                    'department':        p.department,
                    'consultation_type': p.consultation_type,
                    'consultation_date': (
                        p.consultation_date.strftime('%Y-%m-%d')
                        if p.consultation_date else ''
                    ),
                    'consultation_time': (
                        p.consultation_time.strftime('%H:%M')
                        if p.consultation_time else ''
                    ),
                    'queue_number':      p.queue_number or '',
                    'status':            p.status or 'waiting',
                    'check_in_time':     (
                        p.check_in_time.strftime('%Y-%m-%d %H:%M')
                        if p.check_in_time else ''
                    ),
                },
                'confidence': {
                    'cosine_similarity':  round(best_cos,  4),
                    'euclidean_distance': round(best_dist, 4),
                    'similarity_gap':     round(gap,       4),
                    'is_excellent_match': excellent,
                },
                'matching_thresholds': {
                    'cosine':    COSINE_TH,
                    'euclidean': EUCLIDEAN_TH,
                    'gap':       GAP_TH,
                }
            })
        else:
            return jsonify({
                'found': False,
                'best_similarity': round(best_cos,  4),
                'best_distance':   round(best_dist, 4),
                'similarity_gap':  round(gap,       4),
                'matching_thresholds': {
                    'cosine':    COSINE_TH,
                    'euclidean': EUCLIDEAN_TH,
                    'gap':       GAP_TH,
                },
                'message': 'No matching patient found – please proceed with new registration.'
            })

    except Exception as e:
        print(f"[FACE MATCH EXCEPTION] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/queue')
def queue_status():
    """Queue status page"""
    departments = Department.query.all()
    
    content = '''
    <div class="row">
        <div class="col-12">
            <h2 class="mb-4"><i class="fas fa-list-ol me-2"></i>Queue Status</h2>
            <p class="text-muted mb-4">Real-time queue information for all departments</p>
        </div>
    </div>
    
    <div class="row">
    '''
    
    for dept in departments:
        patients = Patient.query.filter_by(department=dept.name).filter(
            Patient.status.in_(['waiting', 'emergency', 'in_progress'])
        ).order_by(
            db.case(
                (Patient.status == 'emergency', 1),
                (Patient.status == 'in_progress', 2),
                (Patient.status == 'waiting', 3)
            ),
            Patient.check_in_time
        ).all()
        
        waiting_count = len([p for p in patients if p.status == 'waiting'])
        emergency_count = len([p for p in patients if p.status == 'emergency'])
        in_progress_count = len([p for p in patients if p.status == 'in_progress'])
        
        content += f'''
        <div class="col-12 col-md-6 col-lg-4 mb-4">
            <div class="card h-100">
                <div class="card-header bg-white">
                    <h5 class="mb-0">
                        <i class="{dept.icon} me-2"></i>{dept.name}
                    </h5>
                </div>
                <div class="card-body">
                    <div class="queue-info">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <div>
                                <h6 class="mb-0">Waiting Patients</h6>
                                <h2 class="text-primary">{waiting_count}</h2>
                            </div>
                            <div class="text-end">
                                <span class="badge bg-danger">Emergency: {emergency_count}</span>
                                <span class="badge bg-info mt-1">In Progress: {in_progress_count}</span>
                            </div>
                        </div>
                        
                        <h6 class="mt-3 mb-2">Current Queue:</h6>
                        <div class="list-group">
        '''
        
        if patients:
            for patient in patients[:5]:  # Show first 5 patients
                status_class = {
                    'emergency': 'danger',
                    'in_progress': 'info',
                    'waiting': 'warning'
                }.get(patient.status, 'secondary')
                
                status_text = patient.status.upper().replace('_', ' ')
                
                content += f'''
                <div class="list-group-item list-group-item-action">
                    <div class="d-flex w-100 justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1">#{patient.queue_number}</h6>
                        </div>
                        <span class="badge bg-{status_class}">{status_text}</span>
                    </div>
                    <small class="text-muted">Check-in: {patient.check_in_time.strftime("%I:%M %p")}</small>
                </div>
            '''
            
            if len(patients) > 5:
                content += f'''
                <div class="list-group-item text-center text-muted">
                    <small>+{len(patients) - 5} more patients</small>
                </div>
                '''
        else:
            content += '''
                <div class="list-group-item">
                    <p class="text-muted mb-0 text-center">No patients in queue</p>
                </div>
            '''
        
        content += f'''
                        </div>
                    </div>
                </div>
                <div class="card-footer bg-white">
                    <small class="text-muted">
                        <i class="fas fa-clock me-1"></i>
                        Est. wait time: {waiting_count * 15} mins
                    </small>
                </div>
            </div>
        </div>
        '''
    
    content += '''
    </div>
    
    <!-- Queue Search for Mobile -->
    <div class="row d-md-none mt-3">
        <div class="col-12">
            <div class="card">
                <div class="card-body">
                    <h6>Check Your Position</h6>
                    <div class="input-group">
                        <input type="text" class="form-control" placeholder="Enter Patient ID" id="queueSearch">
                        <button class="btn btn-primary" onclick="searchQueue()">Check</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function searchQueue() {
            const patientId = document.getElementById('queueSearch').value;
            if (patientId) {
                if (window.localizeHospitalAlert) {{ window.localizeHospitalAlert('queue_help'); }} else {{ alert('Please check the registration desk for your queue status, or use the chatbot for assistance.'); }}
            }
        }
    </script>
    '''
    
    return get_base_html("Queue Status", content, show_chat=True)

@app.route('/navigation')
def navigation():
    """Hospital navigation page"""
    departments = Department.query.all()
    
    # Facilities list
    facilities = [
        {"name": "Cafeteria", "location": "Basement", "icon": "fas fa-utensils", "floor": "B"},
        {"name": "Pharmacy", "location": "Ground Floor", "icon": "fas fa-pills", "floor": "G"},
        {"name": "Billing", "location": "Ground Floor", "icon": "fas fa-file-invoice-dollar", "floor": "G"},
        {"name": "Laboratory", "location": "Ground Floor", "icon": "fas fa-flask", "floor": "G"},
        {"name": "ICU", "location": "First Floor", "icon": "fas fa-heartbeat", "floor": "1"},
        {"name": "Operation Theater", "location": "First Floor", "icon": "fas fa-procedures", "floor": "1"},
        {"name": "Radiology", "location": "Second Floor", "icon": "fas fa-x-ray", "floor": "2"},
        {"name": "Parking", "location": "Basement", "icon": "fas fa-parking", "floor": "B"}
    ]
    
    content = '''
    <div class="row">
        <div class="col-12">
            <h2 class="mb-4"><i class="fas fa-map-signs me-2"></i>Hospital Navigation</h2>
            <p class="text-muted mb-4">Find your way around the hospital</p>
        </div>
    </div>
    
    <!-- Quick Floor Selector for Mobile -->
    <div class="row d-md-none mb-3">
        <div class="col-12">
            <div class="btn-group w-100" role="group">
                <button class="btn btn-outline-primary" onclick="filterFloor('all')">All</button>
                <button class="btn btn-outline-primary" onclick="filterFloor('B')">B</button>
                <button class="btn btn-outline-primary" onclick="filterFloor('G')">G</button>
                <button class="btn btn-outline-primary" onclick="filterFloor('1')">1</button>
                <button class="btn btn-outline-primary" onclick="filterFloor('2')">2</button>
            </div>
        </div>
    </div>
    
        <!-- Map Grid - Responsive -->
    <div class="row mb-4">
        <div class="col-12">
            <div class="card">
                <div class="card-header bg-white">
                    <h5 class="mb-0"><i class="fas fa-map me-2"></i> Hospital Map</h5>
                </div>
                <div class="card-body p-2 p-md-3">
                    <div class="hospital-map-container">
                        <div class="hospital-map p-2 p-md-4" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;">
                            

                            <!-- Second Floor -->
                            <div class="map-floor mb-3 mb-md-4" data-floor="2">
                                <h5 class="text-white mb-3">Second Floor</h5>
                                <div class="row g-2">
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-success text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Orthopedics')">
                                            <i class="fas fa-bone fa-3x mb-2"></i>
                                            <h6>Orthopedics</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-success text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Cardiology')">
                                            <i class="fas fa-heart fa-3x mb-2"></i>
                                            <h6>Cardiology</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-danger text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('ICU 2')">
                                            <i class="fas fa-heartbeat fa-3x mb-2"></i>
                                            <h6>ICU 2</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-4">
                                        <div class="map-room bg-success text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Wards')">
                                            <i class="fas fa-bed fa-3x mb-2"></i>
                                            <h6>Wards</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-1">
                                        <div class="map-room bg-info text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Elevators')">
                                            <i class="fas fa-elevator fa-3x mb-2"></i>
                                            <h6>Lifts</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-1">
                                        <div class="map-room bg-info text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Stairs')">
                                            <i class="fas fa-stairs fa-3x mb-2"></i>
                                            <h6>Stairs</h6>
                                        </div>
                                    </div>
                                </div>
                            </div>
                               
                            <!-- First Floor -->
                            <div class="map-floor mb-3 mb-md-4" data-floor="1">
                                <h5 class="text-white mb-3">First Floor</h5>
                                <div class="row g-2">
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-success text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Gynaecology')">
                                            <i class="fas fa-female fa-3x mb-2"></i>
                                            <h6>Gynaecology</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-success text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Pediatrics')">
                                            <i class="fas fa-baby fa-3x mb-2"></i>
                                            <h6>Pediatrics</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-danger text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('ICU')">
                                            <i class="fas fa-heartbeat fa-3x mb-2"></i>
                                            <h6>ICU</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-danger text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Operation Theater')">
                                            <i class="fas fa-procedures fa-3x mb-2"></i>
                                            <h6>OT</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-success text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Wards')">
                                            <i class="fas fa-bed fa-3x mb-2"></i>
                                            <h6>Wards</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-1">
                                        <div class="map-room bg-info text-white p-3 p-md-1 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Elevators')">
                                            <i class="fas fa-elevator fa-3x mb-2"></i>
                                            <h6>Lifts</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-1">
                                        <div class="map-room bg-info text-white p-3 p-md-1 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Stairs')">
                                            <i class="fas fa-stairs fa-3x mb-2"></i>
                                            <h6>Stairs</h6>
                                        </div>
                                    </div>
                                </div>
                            </div>
                                                        
                            <!-- Ground Floor -->
                            <div class="map-floor mb-3 mb-md-4" data-floor="G">
                                <h5 class="text-white mb-3">Ground Floor</h5>
                                <div class="row g-2">
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-primary text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Main Reception')">
                                            <i class="fas fa-concierge-bell fa-3x mb-2"></i>
                                            <h6>Reception</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-primary text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Billing')">
                                            <i class="fas fa-file-invoice-dollar fa-3x mb-2"></i>
                                            <h6>Billing</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-danger text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Emergency')">
                                            <i class="fas fa-ambulance fa-3x mb-2"></i>
                                            <h6>Emergency</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-success text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Pharmacy')">
                                            <i class="fas fa-pills fa-3x mb-2"></i>
                                            <h6>Pharmacy</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-2">
                                        <div class="map-room bg-success text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Laboratory')">
                                            <i class="fas fa-flask fa-3x mb-2"></i>
                                            <h6>Lab</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-1">
                                        <div class="map-room bg-info text-white p-3 p-md-1 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Elevators')">
                                            <i class="fas fa-elevator fa-3x mb-2"></i>
                                            <h6>Lifts</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-1">
                                        <div class="map-room bg-info text-white p-3 p-md-1 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Stairs')">
                                            <i class="fas fa-stairs fa-3x mb-2"></i>
                                            <h6>Stairs</h6>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Basement -->
                            <div class="map-floor" data-floor="B">
                                <h5 class="text-white mb-3">Basement</h5>
                                <div class="row g-2">
                                    <div class="col-12 col-sm-6 col-lg-6">
                                        <div class="map-room bg-secondary text-white p-3 p-md-6 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Parking')">
                                            <i class="fas fa-parking fa-3x mb-2"></i>
                                            <h6>Parking</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-4">
                                        <div class="map-room bg-warning text-white p-3 p-md-4 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Cafeteria')">
                                            <i class="fas fa-utensils fa-3x mb-2"></i>
                                            <h6>Cafeteria</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-1">
                                        <div class="map-room bg-info text-white p-3 p-md-1 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Elevators')">
                                            <i class="fas fa-elevator fa-3x mb-2"></i>
                                            <h6>Lifts</h6>
                                        </div>
                                    </div>
                                    <div class="col-12 col-sm-6 col-lg-1">
                                        <div class="map-room bg-info text-white p-3 p-md-1 rounded text-center h-100 d-flex flex-column justify-content-center" onclick="showLocation('Stairs')">
                                            <i class="fas fa-stairs fa-3x mb-2"></i>
                                            <h6>Stairs</h6>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    
    <!-- Departments and Facilities Lists -->
    <div class="row">
        <div class="col-12 col-md-6 mb-4">
            <div class="card h-100">
                <div class="card-header bg-white">
                    <h5 class="mb-0"><i class="fas fa-clinic-medical me-2"></i>Medical Departments</h5>
                </div>
                <div class="card-body p-0">
                    <div class="list-group list-group-flush">
    '''
    
    for dept in departments:
        content += f'''
                        <div class="list-group-item list-group-item-action" onclick="showLocation('{dept.name}')">
                            <div class="d-flex justify-content-between align-items-center">
                                <div class="d-flex align-items-center">
                                    <i class="{dept.icon} me-3 text-primary"></i>
                                    <div>
                                        <h6 class="mb-0">{dept.name}</h6>
                                        <small class="text-muted d-none d-md-block">{dept.location}</small>
                                    </div>
                                </div>
                                <span class="badge bg-primary">Floor {dept.floor}</span>
                            </div>
                        </div>
        '''
    
    content += '''
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-12 col-md-6 mb-4">
            <div class="card h-100">
                <div class="card-header bg-white">
                    <h5 class="mb-0"><i class="fas fa-building me-2"></i>Facilities</h5>
                </div>
                <div class="card-body p-0">
                    <div class="list-group list-group-flush">
    '''
    
    for facility in facilities:
        content += f'''
                        <div class="list-group-item list-group-item-action" onclick="showLocation('{facility['name']}')">
                            <div class="d-flex justify-content-between align-items-center">
                                <div class="d-flex align-items-center">
                                    <i class="{facility['icon']} me-3 text-success"></i>
                                    <div>
                                        <h6 class="mb-0">{facility['name']}</h6>
                                        <small class="text-muted d-none d-md-block">{facility['location']}</small>
                                    </div>
                                </div>
                                <span class="badge bg-success">Floor {facility['floor']}</span>
                            </div>
                        </div>
        '''
    
    content += '''
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function showLocation(location) {
            if (window.localizeHospitalAlert) {{ window.localizeHospitalAlert('location_help', {{ location }}); }}
            else {{ alert(`📍 ${location}\n\nPlease follow the signs or ask staff for directions. Use the chatbot for more details.`); }}
        }
        
        function filterFloor(floor) {
            const floors = document.querySelectorAll('.map-floor');
            floors.forEach(f => {
                if (floor === 'all' || f.getAttribute('data-floor') === floor) {
                    f.style.display = 'block';
                } else {
                    f.style.display = 'none';
                }
            });
        }
    </script>
    '''
    
    return get_base_html("Hospital Navigation", content, show_chat=True)

@app.route('/emergency')
def emergency():
    """Emergency page"""
    content = '''
    <div class="row justify-content-center">
        <div class="col-12 col-lg-8">
            <div class="alert alert-danger text-center py-4 py-md-5">
                <i class="fas fa-ambulance fa-4x mb-3"></i>
                <h1 class="alert-heading display-4">EMERGENCY</h1>
                <p class="lead">If this is a medical emergency, please take immediate action:</p>
                
                <div class="row mt-4 g-3">
                    <div class="col-12 col-md-6">
                        <div class="card border-danger h-100">
                            <div class="card-body text-center py-4">
                                <i class="fas fa-phone fa-3x text-danger mb-3"></i>
                                <h3>Call Emergency</h3>
                                <a href="tel:112" class="display-1 text-danger my-3 d-inline-block text-decoration-none" aria-label="Call 112">112</a>
                                <p class="mb-0">Toll-free, 24/7</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-12 col-md-6">
                        <div class="card border-danger h-100">
                            <div class="card-body text-center py-4">
                                <i class="fas fa-hospital fa-3x text-danger mb-3"></i>
                                <h3>Emergency Department</h3>
                                <div class="my-3">
                                    <h5>Main Building</h5>
                                    <p>Ground Floor, Wing A</p>
                                    <p class="mb-0">Open 24/7</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-4">
                    <a href="/" class="btn btn-outline-light btn-lg">Return to Home</a>
                </div>
            </div>
        </div>
    </div>
    '''
    
    return get_base_html("Emergency Assistance", content, show_chat=True)

@app.route('/api/department/<dept_name>/doctors')
def get_doctors_by_department(dept_name):
    """API endpoint to get doctors by department"""
    doctors = User.query.filter_by(department=dept_name, role='doctor').all()
    
    if not doctors:
        return jsonify([])
    
    return jsonify([{
        'id': doctor.id,
        'name': doctor.full_name,
        'username': doctor.username,
        'department': doctor.department
    } for doctor in doctors])

@app.route('/api/train-face-model', methods=['POST'])
def train_face_model_endpoint():
    """Train the face recognition model"""
    try:
        result = train_face_model()
        if result:
            return jsonify({
                'success': True,
                'message': 'Face model trained successfully',
                'config': {
                    'cosine_threshold': result['cosine_threshold'],
                    'euclidean_threshold': result['euclidean_threshold'],
                    'total_patients': result['total_patients']
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Training failed'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Initialize the database
init_db()
if sms_is_configured():
    print("[OK] SMS notifications configured")
else:
    print("[!] SMS notifications not configured – set Twilio env vars or a local .env file")

# Main execution
if __name__ == '__main__':
    print("=" * 60)
    print("SMART HOSPITAL KIOSK ")
    print("=" * 60)
    print("[OK] Application is running!")
    print("Open in your browser: http://localhost:5000")
    print("\nFeatures:")
    print("   * Responsive Design (Mobile & Desktop)")
    print("   * AI Chatbot Assistant")
    print("   * Patient Registration")
    print("   * Doctor Dashboard")
    print("   * Queue Management")
    print("   * Hospital Navigation")
    print("\nDoctor Credentials:")
    print('''==========================================================================================
HOSPITAL SYSTEM - ALL USER CREDENTIALS
==========================================================================================

[ADMIN ACCOUNT]
------------------------------------------------------------------------------------------
  Username: admin                     | Password: admin123

[CARDIOLOGY]
------------------------------------------------------------------------------------------
  Dr. John Smith                      | dr_smith             | doctor123
  Dr. Robert Miller                   | dr_miller            | doctor123
  Dr. Lisa Chen                       | dr_chen              | doctor123

[EMERGENCY]
------------------------------------------------------------------------------------------
  Dr. Sarah Jones                     | dr_jones             | doctor123
  Dr. Rajesh Kumar                    | dr_kumar             | doctor123

[GYNAECOLOGY]
------------------------------------------------------------------------------------------
  Dr. Emily Wang                      | dr_wang              | doctor123
  Dr. Priya Sharma                    | dr_sharma            | doctor123
  Dr. Maria Jackson                   | dr_jackson           | doctor123

[ORTHOPEDICS]
------------------------------------------------------------------------------------------
  Dr. Raj Patel                       | dr_patel             | doctor123
  Dr. Carlos Rodriguez                | dr_rodriguez         | doctor123

[PEDIATRICS]
------------------------------------------------------------------------------------------
  Dr. David Anderson                  | dr_anderson          | doctor123
  Dr. Anjali Gupta                    | dr_gupta             | doctor123

[PHARMACY]
------------------------------------------------------------------------------------------
  Ms. Jennifer Lee                    | pharm_lee            | doctor123
  Mr. James Wilson                    | pharm_wilson         | doctor123

==========================================================================================
TOTAL USERS: 15 (1 Admin + 14 Doctors/Staff)
==========================================================================================''')

    print("=" * 60)

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
    
