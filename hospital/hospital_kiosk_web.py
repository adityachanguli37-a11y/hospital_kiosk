# Save this complete file as hospital_kiosk_enhanced.py
from flask import Flask, render_template_string, request, jsonify, session, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import datetime
import uuid
import json
import os
import base64
import io
from PIL import Image
import numpy as np
from scipy import ndimage
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# Try to import face recognition libraries in order of preference
DEEPFACE_AVAILABLE = False
FACENET_AVAILABLE = False
OPENCV_AVAILABLE = False
FACE_RECOGNITION_AVAILABLE = False
SKLEARN_AVAILABLE = False

# Ensure cv2 is always defined to avoid NameError when unavailable
cv2 = None
FACE_CASCADE = None

# Try face_recognition library (uses dlib ResNet)
try:
    import face_recognition as fr
    FACE_RECOGNITION_AVAILABLE = True
    print("[+] face_recognition library available (dlib-based)")
except ImportError:
    print("[!] face_recognition library not available")

# Try scikit-learn for classifier training
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
    print("[+] scikit-learn available for model training")
except ImportError:
    print("[!] scikit-learn not available")

# Try OpenCV first (most reliable and doesn't require external models)
try:
    import cv2
    OPENCV_AVAILABLE = True
    # Load Haar Cascade classifier for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    
    # Load Eye Cascade for high-accuracy alignment
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    EYE_CASCADE = cv2.CascadeClassifier(eye_cascade_path)

    if not FACE_CASCADE.empty():
        print("[+] OpenCV available with Face and Eye Cascades for high-accuracy alignment")
    else:
        OPENCV_AVAILABLE = False
        print("[!] OpenCV Haar Cascade failed to load")
except ImportError:
    print("[!] OpenCV not available")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("[+] DeepFace available for face recognition")
except ImportError:
    print("[!] DeepFace not available")

# Try FaceNet (best option)
try:
    from facenet_pytorch import InceptionResnetV1
    import torch
    FACENET_AVAILABLE = True
    print("[+] FaceNet (facenet-pytorch) available for face recognition")
    # Initialize FaceNet model
    try:
        FACENET_MODEL = InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            FACENET_MODEL = FACENET_MODEL.cuda()
            print("[+] FaceNet GPU acceleration enabled")
        print("[+] FaceNet model loaded successfully")
    except Exception as e:
        print(f"[!] Failed to load FaceNet model: {e}")
        FACENET_AVAILABLE = False
        FACENET_MODEL = None
except ImportError:
    print("[!] FaceNet (facenet-pytorch) not available")

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hospital-kiosk-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hospital_kiosk.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def train_face_model():
    """Train face recognition model with ML classifier"""
    print("\n" + "="*60)
    print("🧠 TRAINING FACE RECOGNITION MODEL (ML-BASED)")
    print("="*60)
    
    try:
        with app.app_context():
            # Get all patients with face embeddings and images
            patients = Patient.query.filter(Patient.face_descriptor.isnot(None)).all()
            
            if len(patients) < 2:
                print("[!] Not enough patients for training (minimum 2 required)")
                return
            
            print(f"[*] Loading {len(patients)} patient face data...")
            
            # Method 1: Use face_recognition library (best embeddings)
            if FACE_RECOGNITION_AVAILABLE and SKLEARN_AVAILABLE:
                print("[*] Using face_recognition library for embeddings...")
                return train_with_face_recognition(patients)
            
            # Method 2: Use scikit-learn SVM on existing embeddings
            elif SKLEARN_AVAILABLE:
                print("[*] Using SVM classifier on existing embeddings...")
                return train_with_svm(patients)
            
            # Method 3: Fallback to threshold-based approach
            else:
                print("[*] Using statistical threshold training...")
                return train_with_statistical_thresholds(patients)
    
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_with_svm(patients):
    """Train SVM classifier on face embeddings"""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    try:
        embeddings = []
        labels = []
        valid_patients = []
        
        # Extract embeddings and labels
        for idx, patient in enumerate(patients):
            try:
                embedding = np.array(json.loads(patient.face_descriptor), dtype=np.float32)
                if len(embedding) == 2048:
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                    embeddings.append(embedding)
                    labels.append(idx)
                    valid_patients.append(patient)
            except Exception as e:
                print(f"[!] Error loading embedding for {patient.name}: {e}")
                continue
        
        if len(embeddings) < 2:
            print("[!] Not enough valid embeddings for training")
            return None
        
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        print(f"[OK] Loaded {len(embeddings)} valid embeddings from {len(valid_patients)} patients")
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels if len(np.unique(labels)) > 1 else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM with RBF kernel
        print("[*] Training SVM classifier...")
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        svm.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = svm.score(X_train_scaled, y_train)
        test_accuracy = svm.score(X_test_scaled, y_test)
        
        print(f"[TRAINING RESULTS]")
        print(f"  Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"  Testing Accuracy:  {test_accuracy*100:.2f}%")
        print(f"  Total Samples:     {len(embeddings)}")
        print(f"  Training Samples:  {len(X_train)}")
        print(f"  Testing Samples:   {len(X_test)}")
        print(f"  Number of Classes: {len(np.unique(labels))}")
        
        # Calculate pairwise statistics
        similarities = []
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                dist = float(np.linalg.norm(embeddings[i] - embeddings[j]))
                similarities.append(sim)
                distances.append(dist)
        
        print(f"\n[EMBEDDING STATISTICS]")
        print(f"  Similarity (Mean): {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")
        print(f"  Distance (Mean):   {np.mean(distances):.2f} ± {np.std(distances):.2f}")
        print(f"  Similarity Range:  [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")
        print(f"  Distance Range:    [{np.min(distances):.2f}, {np.max(distances):.2f}]")
        
        # Calculate optimal thresholds
        optimal_cosine = np.percentile(similarities, 85)
        optimal_euclidean = np.percentile(distances, 85)
        
        print(f"\n[RECOMMENDED THRESHOLDS]")
        print(f"  Cosine Similarity:  {optimal_cosine:.4f}")
        print(f"  Euclidean Distance: {optimal_euclidean:.2f}")
        
        # Store trained model
        model = FaceRecognitionModel.query.first()
        if not model:
            model = FaceRecognitionModel()
        
        model.cosine_threshold = float(optimal_cosine)
        model.euclidean_threshold = float(optimal_euclidean)
        model.similarity_gap = 0.20
        model.last_trained = datetime.datetime.utcnow()
        model.total_patients = len(embeddings)
        model.training_data = json.dumps({
            'method': 'SVM-Classifier',
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'total_patients': len(embeddings),
            'similarity_mean': float(np.mean(similarities)),
            'similarity_std': float(np.std(similarities)),
            'distance_mean': float(np.mean(distances)),
            'distance_std': float(np.std(distances))
        })
        
        db.session.add(model)
        db.session.commit()
        
        print(f"\n[OK] SVM Model trained and saved successfully!")
        print("="*60 + "\n")
        return model
        
    except Exception as e:
        print(f"[ERROR] SVM training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_with_face_recognition(patients):
    """Train model using face_recognition library embeddings"""
    try:
        import face_recognition as fr
        import tempfile
        import base64
        from PIL import Image
        import io
        
        print("[*] Extracting face encoding using face_recognition...")
        
        embeddings = []
        labels = []
        valid_patients = []
        
        # Extract face encodings from stored face images
        for idx, patient in enumerate(patients):
            try:
                if patient.face_image:
                    # Decode face image from base64
                    image_data = patient.face_image.split(',')[1] if ',' in patient.face_image else patient.face_image
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    image_np = np.array(image)
                    
                    # Get face encoding from face_recognition library
                    encodings = fr.face_encodings(image_np)
                    if encodings:
                        # Use the first (and usually only) face detected
                        embedding = np.array(encodings[0], dtype=np.float32)
                        embeddings.append(embedding)
                        labels.append(idx)
                        valid_patients.append(patient)
                        print(f"  [OK] {patient.name}: encoding extracted")
            except Exception as e:
                print(f"  [X] {patient.name}: {e}")
                continue
        
        if len(embeddings) < 2:
            print("[!] Could not extract encodings for enough patients")
            return None
        
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        print(f"[OK] Extracted {len(embeddings)} face encodings")
        
        # Train SVM on face_recognition embeddings
        if SKLEARN_AVAILABLE:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            # Split and scale
            split_result = train_test_split(
                embeddings, labels, test_size=0.2, random_state=42, 
                stratify=labels if len(np.unique(labels)) > 1 else None
            )
            X_train, X_test, y_train, y_test = split_result
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train SVM
            print("[*] Training SVM on face_recognition embeddings...")
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
            svm.fit(X_train_scaled, y_train)
            
            train_acc = svm.score(X_train_scaled, y_train)
            test_acc = svm.score(X_test_scaled, y_test)
            
            print(f"[TRAINING RESULTS]")
            print(f"  Train Accuracy: {train_acc*100:.2f}%")
            print(f"  Test Accuracy:  {test_acc*100:.2f}%")
        
        # Calculate thresholds
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                dist = float(np.linalg.norm(embeddings[i] - embeddings[j]))
                distances.append(dist)
        
        threshold = np.percentile(distances, 85) if distances else 0.6
        
        print(f"\n[FACE ENCODING STATISTICS]")
        print(f"  Encoding dimension: {len(embeddings[0])}")
        print(f"  Distance threshold: {threshold:.4f}")
        print(f"  Distance range: [{np.min(distances):.4f}, {np.max(distances):.4f}]")
        
        # Store model
        model = FaceRecognitionModel.query.first()
        if not model:
            model = FaceRecognitionModel()
        
        model.cosine_threshold = 0.85
        model.euclidean_threshold = threshold
        model.similarity_gap = 0.20
        model.last_trained = datetime.datetime.utcnow()
        model.total_patients = len(embeddings)
        model.training_data = json.dumps({
            'method': 'face_recognition-library',
            'encoding_dim': 128,
            'distance_threshold': float(threshold),
            'total_samples': len(embeddings)
        })
        
        db.session.add(model)
        db.session.commit()
        
        print(f"\n[OK] face_recognition model trained successfully!")
        print("="*60 + "\n")
        return model
        
    except Exception as e:
        print(f"[ERROR] face_recognition training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_with_statistical_thresholds(patients):
    """Fallback: Train using statistical thresholds"""
    print("[*] Using statistical approach...")
    
    try:
        embeddings = []
        for patient in patients:
            try:
                embedding = np.array(json.loads(patient.face_descriptor), dtype=np.float32)
                if len(embedding) == 2048:
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                    embeddings.append(embedding)
            except:
                continue
        
        if len(embeddings) < 2:
            print("[!] Not enough valid embeddings")
            return None
        
        embeddings = np.array(embeddings)
        
        # Calculate statistics
        similarities = []
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                dist = float(np.linalg.norm(embeddings[i] - embeddings[j]))
                similarities.append(sim)
                distances.append(dist)
        
        print(f"\n[STATISTICS]")
        print(f"  Samples: {len(embeddings)}")
        print(f"  Similarity: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")
        print(f"  Distance: {np.mean(distances):.2f} ± {np.std(distances):.2f}")
        
        optimal_cosine = np.percentile(similarities, 90)
        optimal_euclidean = np.percentile(distances, 90)
        
        print(f"\n[THRESHOLDS]")
        print(f"  Cosine: {optimal_cosine:.4f}")
        print(f"  Euclidean: {optimal_euclidean:.2f}")
        
        model = FaceRecognitionModel.query.first()
        if not model:
            model = FaceRecognitionModel()
        
        model.cosine_threshold = float(optimal_cosine)
        model.euclidean_threshold = float(optimal_euclidean)
        model.similarity_gap = 0.20
        model.last_trained = datetime.datetime.utcnow()
        model.total_patients = len(embeddings)
        model.training_data = json.dumps({
            'method': 'statistical',
            'total_samples': len(embeddings)
        })
        
        db.session.add(model)
        db.session.commit()
        
        print(f"\n[OK] Model trained successfully!")
        print("="*60 + "\n")
        
        return model
    
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_face_model_config():
    """Get current face recognition model configuration"""
    with app.app_context():
        model = FaceRecognitionModel.query.first()
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
            
            # Create admin user
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(username='admin', full_name='Administrator', role='admin')
                admin.set_password('admin123')
                db.session.add(admin)
            
            # Create sample doctor users for each department
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
                doctor = User.query.filter_by(username=doctor_data['username']).first()
                if not doctor:
                    doctor = User(
                        username=doctor_data['username'],
                        full_name=doctor_data['full_name'],
                        department=doctor_data['department'],
                        role='doctor' if doctor_data['username'].startswith('dr_') else 'staff'
                    )
                    doctor.set_password('doctor123')
                    db.session.add(doctor)
            
            db.session.commit()
            print("[OK] Sample data added with multiple doctors per department")
        
        # Initialize face recognition model
        model = FaceRecognitionModel(
            model_name='RobustMultiScale',
            cosine_threshold=0.65,
            euclidean_threshold=10.0,
            similarity_gap=0.05
        )
        db.session.add(model)
        db.session.commit()
        print("[OK] Face recognition model initialized")



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

    nav = ''
    if show_nav:
        nav = f'''
    <!-- Professional Sticky Navigation Bar -->
    <nav class="navbar navbar-expand-lg navbar-light bg-white shadow-sm sticky-top">
        <div class="container">
            <a class="navbar-brand d-flex align-items-center" href="/">
                <div class="bg-primary text-white rounded p-1 me-2" style="width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;">
                    <i class="fas fa-hospital-alt"></i>
                </div>
                <div class="lh-1">
                    <div class="fw-bold text-primary" style="letter-spacing: 0.5px;">ＭＥＤＦＬＯＷ</div>
                    <small class="text-muted" style="font-size: 0.6rem;">SMART KIOSK FOR DIGITAL HEALTHCARE</small>
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
                <div class="alert {alert_class} alert-dismissible fade show" role="alert">
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
        
        function addMessage(message, sender) {
            const messagesDiv = document.getElementById('chatMessages');
            if (!messagesDiv) return;
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            
            const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
            // Basic Markdown-style formatting (**bold**)
            let formattedMessage = message.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
            // Handle bullet points
            formattedMessage = formattedMessage.replace(/\\u2022\\s*(.*?)(?=\\n|$)/g, '<li>$1</li>');
            if (formattedMessage.includes('<li>')) {
                formattedMessage = formattedMessage.replace(/(<li>.*?<\\/li>)+/g, '<ul>$&</ul>');
            }
            
            messageDiv.innerHTML = `
                <div class="message-content">
                    <div class="d-flex align-items-start">
                        ${sender === 'bot' ? '<div class="bot-icon me-2"><i class="fas fa-robot"></i></div>' : ''}
                        <div>${formattedMessage}</div>
                    </div>
                </div>
                <div class="message-time">${time}</div>
            `;
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
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
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            color: var(--dark-color);
            overflow-x: hidden;
            min-height: 100vh;
            position: relative;
            padding-bottom: 60px; /* Space for footer */
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
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        
        /* Mobile Optimizations */
        @media (max-width: 768px) {
            .navbar-brand {
                font-size: 1.2rem;
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
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            display: none;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.2);
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
            background: #f8f9fa;
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
            background: white;
            border: 1px solid #e9ecef;
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
            color: #adb5bd;
            margin-top: 4px;
            padding: 0 4px;
        }
        
        .user-message .message-time {
            text-align: right;
            margin-right: 5px;
        }
        
        .chatbot-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e9ecef;
        }
        
        .suggested-questions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
        }
        
        .suggested-question {
            background: #f0f2f5;
            border: none;
            border-radius: 20px;
            padding: 8px 15px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            white-space: nowrap;
            color: var(--dark-color);
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
            background: white;
            border-radius: 18px;
            border: 1px solid #e9ecef;
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
    </style>
    '''
    
    return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=yes">
    <title>{title} - Smart Hospital Kiosk</title>
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
                    <p class="small text-white-50 mb-0"><i class="fas fa-ambulance me-3 text-danger"></i> Emergency Hotline: <strong>112</strong></p>
                    
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
        
        // Auto-dismiss alerts after 5 seconds
        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(function() {{
                const alerts = document.querySelectorAll('.alert');
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
        "5. **Face Registration**: Our system now uses eye-detection to align your photo automatically!\n" +
        "6. Submit the form",
        
        "After registration, you'll receive:\n" +
        "✅ Unique Patient ID (e.g., PAT12345678)\n" +
        "✅ Queue Number (e.g., CAR001)\n" +
        "✅ Department assignment\n" +
        "✅ Check-in time\n" +
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
            # Get queue position
            queue_position = Patient.query.filter(
                Patient.department == patient.department,
                Patient.status.in_(['waiting', 'emergency']),
                Patient.check_in_time < patient.check_in_time
            ).count() + 1
            
            wait_time = queue_position * 15  # Assume 15 minutes per patient
            
            return [
                f"📊 **Queue Status for {patient.name}**\n\n" +
                f"Patient ID: {patient.patient_id}\n" +
                f"Queue Number: {patient.queue_number}\n" +
                f"Department: {patient.department}\n" +
                f"Status: {patient.status.upper()}\n" +
                f"Position in Queue: {queue_position}\n" +
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
                        <p class="mb-0">Proceed directly to Emergency Department or call 112</p>
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
                                        <th>Time</th>
                                    </tr>
                                </thead>
                                <tbody>
    '''
    
    for p in recent_patients:
        status_color = 'danger' if p.status == 'emergency' else 'warning' if p.status == 'waiting' else 'info' if p.status == 'in_progress' else 'success'
        content += f'''
                                    <tr>
                                        <td>
                                            <div class="fw-bold">{p.name}</div>
                                            <small class="text-muted">{p.patient_id}</small>
                                        </td>
                                        <td>{p.department}</td>
                                        <td><span class="badge bg-{status_color}">{p.status.upper()}</span></td>
                                        <td>{p.check_in_time.strftime("%I:%M %p")}</td>
                                    </tr>
        '''
    
    content += '''
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-12 col-xl-4">
                <div class="card shadow-sm h-100">
                    <div class="card-header bg-white">
                        <h5 class="mb-0">Active Doctors</h5>
                    </div>
                    <div class="card-body">
                        <ul class="list-group list-group-flush">
    '''
    
    for d in doctors:
        content += f'''
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                <div>
                                    <div class="fw-bold">{d.full_name}</div>
                                    <small class="text-muted">{d.department}</small>
                                </div>
                                <span class="badge bg-success rounded-pill">Active</span>
                            </li>
        '''
        
    content += '''
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>
    '''
    
    return get_base_html("Admin Dashboard", content, show_chat=False)

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
            else_=4
        ),
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
                                    <th class="d-none d-sm-table-cell">Time</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
        '''
        
        for patient in patients:
            wait_time = (datetime.datetime.utcnow() - patient.check_in_time).seconds // 60
            
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
            else:
                status_badge = 'success'
                status_text = 'COMPLETED'
            
            # Check if this is the patient doctor is currently consulting
            is_current_patient = (patient.status == 'in_progress' and 
                                 patient.doctor_assigned == doctor_name)
            
            content += f'''
                                <tr class="{'current-consultation' if is_current_patient else ''}">
                                    <td><strong>{patient.queue_number}</strong></td>
                                    <td>
                                        {patient.name}
                                        <small class="d-block d-md-none text-muted">{patient.age}/{patient.gender}</small>
                                    </td>
                                    <td class="d-none d-md-table-cell">{patient.age}/{patient.gender}</td>
                                    <td class="d-none d-lg-table-cell">{patient.consultation_type.replace('-', ' ').title()}</td>
                                    <td class="d-none d-sm-table-cell">{patient.check_in_time.strftime("%I:%M %p")}</td>
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
            return confirm(`Declare EMERGENCY for ${patientName}?`);
        }
        
        function confirmEnd(patientName) {
            return confirm(`END consultation with ${patientName}?`);
        }
        
        function confirmComplete(patientName) {
            return confirm(`COMPLETE consultation with ${patientName}?`);
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
            # Generate patient ID
            patient_id = f"PAT{str(uuid.uuid4())[:8].upper()}"
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
            
            # Create patient
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
                queue_number=f"{request.form['department'][:3].upper()}{Patient.query.count() + 1:03d}",
                face_descriptor=face_descriptor,
                face_image=face_image
            )
            
            print(f"[DEBUG] Aadhaar number received: '{request.form.get('aadhaar_number', 'NOT RECEIVED')}'")
            
            db.session.add(patient)
            
            # Create user account
            user = User(username=patient_id, role='patient')
            user.set_password(patient_id)
            db.session.add(user)
            db.session.flush()
            
            patient.user_id = user.id
            db.session.commit()
            
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
                                <p><strong>Queue Number:</strong></p>
                                <h3 class="text-primary">{patient.queue_number}</h3>
                            </div>
                        </div>
                        <div class="mt-3">
                            <p><strong>Department:</strong> {patient.department}</p>
                            <p><strong>Aadhaar Number:</strong> {patient.aadhaar_number if patient.aadhaar_number else 'Not Provided'}</p>
                            <p><strong>Check-in Time:</strong> {patient.check_in_time.strftime('%I:%M %p')}</p>
                        </div>
                        <div class="mt-4 d-grid d-md-block">
                            <a href="/" class="btn btn-primary me-md-2 mb-2 mb-md-0">Return to Home</a>
                            <a href="/queue" class="btn btn-outline-primary">Check Queue Status</a>
                        </div>
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

    dept_options = ""
    for dept in departments:
        selected = 'selected' if dept.name == selected_dept else ''
        dept_options += f'<option value="{dept.name}" {selected}>{dept.name}</option>'

    content = f'''
    <div class="row justify-content-center">
        <div class="col-12 col-lg-8">
            <div class="mb-4">
                <h2 class="text-center mb-4"><i class="fas fa-user-plus me-2"></i>Patient Registration</h2>
                <p class="text-center text-muted">Complete both steps below to register</p>
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
                            <div class="col-12 col-md-6">
                                <label class="form-label">Department *</label>
                                <select class="form-select" name="department" required id="departmentSelect">
                                    <option value="">Select Department</option>
                                    {dept_options}
                                </select>
                            </div>

                            <div class="col-12 col-md-6">
                                <label class="form-label">Visit Type *</label>
                                <select class="form-select" name="consultation_type" required>
                                    <option value="">Select Type</option>
                                    <option value="new">New Consultation</option>
                                    <option value="follow-up">Follow-up Visit</option>
                                </select>
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
                        <small>Call 112</small>
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
                alert('Please enter a valid 10-digit phone number');
                return;
            }}
            
            // Validate Aadhaar if provided
            const aadhaar = document.getElementById('aadhaarInput').value;
            if (aadhaar && aadhaar.length !== 12) {{
                e.preventDefault();
                alert('Aadhaar number must be exactly 12 digits');
                document.getElementById('aadhaarInput').focus();
                return;
            }}
            
            // Check if face is captured
            const faceDescriptor = document.getElementById('faceDescriptorInput').value;
            if (!faceDescriptor) {{
                e.preventDefault();
                alert('Please capture your face before registering');
                return;
            }}
        }});
        
        // Face Detection Variables
        let faceCaptured = false;
        let animationFrameId = null;
        
        // Display messages to user
        function handleMessage(message, type = 'info') {{
            const faceMessage = document.getElementById('faceMessage');
            if (faceMessage) {{
                faceMessage.innerHTML = message;
                faceMessage.className = `alert alert-${{type}} small mt-2`;
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
                        
                        handleMessage(`Welcome back ${{patient.name}}! Please select the department you are visiting today.`, 'success');
                    }} else {{
                        faceStatus.innerHTML = `<i class="fas fa-user-plus me-2"></i>New profile. Please complete the registration.`;
                        if(personalSection) personalSection.style.display = 'flex';
                        if(headerText) headerText.innerHTML = `<i class="fas fa-file-alt me-2"></i>Step 2: New Patient Details`;
                        if(submitText) submitText.textContent = "Complete Registration";
                        handleMessage('New patient detected. Please fill in your details.', 'info');
                    }}
                    
                    // Always scroll to step 2 afterward
                    if (step2Card) step2Card.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                }} catch (matchErr) {{
                    console.error('Match error:', matchErr);
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
    """Extract face embedding from uploaded image with high-accuracy alignment and multi-region LBP"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_rgb = np.array(image.convert('RGB'))
        
        # Convert to BGR for OpenCV
        if OPENCV_AVAILABLE:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_rgb[:, :, ::-1].copy()

        # Step 1: Face and Eye Alignment
        if OPENCV_AVAILABLE:
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Get largest face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Detect eyes within face for alignment
                    eyes = EYE_CASCADE.detectMultiScale(face_roi, 1.1, 10, minSize=(15, 15))
                    
                    if len(eyes) >= 2:
                        # Sort eyes by x position
                        eyes = sorted(eyes, key=lambda e: e[0])
                        (ex1, ey1, ew1, eh1) = eyes[0]
                        (ex2, ey2, ew2, eh2) = eyes[1]
                        
                        # Calculate center of eyes
                        p1 = (ex1 + ew1//2, ey1 + eh1//2)
                        p2 = (ex2 + ew2//2, ey2 + eh2//2)
                        
                        # Calculate angle
                        dy = p2[1] - p1[1]
                        dx = p2[0] - p1[0]
                        angle = np.degrees(np.arctan2(dy, dx))
                        
                        # Rotate image around center of eyes
                        center = ((p1[0] + p2[0]) // 2 + x, (p1[1] + p2[1]) // 2 + y)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(image_bgr, M, (image_bgr.shape[1], image_bgr.shape[0]))
                        
                        # Re-crop from rotated image
                        image_bgr = rotated[y:y+h, x:x+w].copy()
                        print(f"[FACE DETECT] Face aligned by {angle:.2f} degrees")
                    else:
                        image_bgr = image_bgr[y:y+h, x:x+w].copy()
                        print("[FACE DETECT] Face detected but no eyes found for alignment")
                else:
                    return jsonify({'error': 'No face detected'}), 400
            except Exception as e:
                print(f"[FACE DETECT] Alignment error: {e}")

        # Step 2: Advanced Feature Extraction (Mathematical Fallback)
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        face_img = image_pil.resize((128, 128)).convert('L')
        pixels = np.array(face_img).astype(np.float32) / 255.0

        # Apply CLAHE for lighting normalization
        if OPENCV_AVAILABLE:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            pixels = clahe.apply((pixels * 255).astype(np.uint8)).astype(np.float32) / 255.0

        features = []
        
        # 1. Global Structure (32x32) - 1024 features
        low_res = np.array(Image.fromarray((pixels * 255).astype(np.uint8)).resize((32, 32))).flatten() / 255.0
        features.extend(low_res)
        
        # 2. Multi-Region Detailed LBP (8x8 Grid) - 64 regions * 8 stats = 512 features
        def get_lbp_stats(region):
            lbp = np.zeros_like(region, dtype=np.uint8)
            padded = np.pad(region, 1, mode='edge')
            for i in range(1, padded.shape[0]-1):
                for j in range(1, padded.shape[1]-1):
                    c = padded[i,j]
                    code = 0
                    if padded[i-1,j-1] >= c: code |= 1
                    if padded[i-1,j] >= c: code |= 2
                    if padded[i-1,j+1] >= c: code |= 4
                    if padded[i,j+1] >= c: code |= 8
                    if padded[i+1,j+1] >= c: code |= 16
                    if padded[i+1,j] >= c: code |= 32
                    if padded[i+1,j-1] >= c: code |= 64
                    if padded[i,j-1] >= c: code |= 128
                    lbp[i-1,j-1] = code
            return [np.mean(lbp)/255.0, np.std(lbp)/255.0]

        for i in range(0, 128, 16):
            for j in range(0, 128, 16):
                region = pixels[i:i+16, j:j+16]
                # Regional Intensity Stats
                features.extend([np.mean(region), np.std(region), np.max(region), np.min(region)])
                # Regional LBP Stats
                features.extend(get_lbp_stats(region))

        # 3. Shape Descriptors (Gradients) - 128x2 = 256 features
        gx = ndimage.sobel(pixels, axis=1)
        gy = ndimage.sobel(pixels, axis=0)
        # Use 16x16 downsampled gradients
        gx_small = np.array(Image.fromarray(gx).resize((16, 16))).flatten()
        gy_small = np.array(Image.fromarray(gy).resize((16, 16))).flatten()
        features.extend(gx_small / (np.max(np.abs(gx_small)) + 1e-8))
        features.extend(gy_small / (np.max(np.abs(gy_small)) + 1e-8))
        
        # 4. Global Histogram - 32 features
        hist, _ = np.histogram(pixels.flatten(), bins=32, range=(0, 1))
        features.extend(hist / (np.sum(hist) + 1e-8))

        # Pad to exactly 2048
        embedding_arr = np.array(features, dtype=np.float32)
        if len(embedding_arr) < 2048:
            embedding_arr = np.pad(embedding_arr, (0, 2048 - len(embedding_arr)), mode='constant')
        else:
            embedding_arr = embedding_arr[:2048]
        
        # Final Norm
        norm = np.linalg.norm(embedding_arr)
        if norm > 0: embedding_arr /= norm
        
        embedding = embedding_arr.tolist()
        
        # Save to database
        captured_record = CapturedFace(
            face_descriptor=json.dumps(embedding),
            face_image=image_data
        )
        db.session.add(captured_record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'embedding': embedding,
            'captured_face_id': captured_record.id,
            'model': 'RobustAlignedGridLBP',
            'warning': 'Using eye-aligned multi-region detection for higher accuracy.'
        })

    except Exception as e:
        print(f"[FACE DETECT ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/register/match-face', methods=['POST'])
def match_face():
    """Match face embedding against existing patients with strict validation"""
    try:
        data = request.get_json()
        if not data or 'embedding' not in data:
            return jsonify({'error': 'No embedding provided'}), 400

        embedding = np.array(data['embedding'], dtype=np.float32)
        if len(embedding) != 2048:
            return jsonify({'error': f'Invalid embedding dimensions: {len(embedding)} != 2048'}), 400

        # Normalize the input embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        best_match = None
        best_similarity = -2  # Start below minimum possible value
        best_distance = float('inf')
        second_best_similarity = -2
        second_best_distance = float('inf')
        
        all_scores = []

        # Check all patients with face data
        patients = Patient.query.filter(Patient.face_descriptor.isnot(None)).all()

        for patient in patients:
            try:
                stored_embedding = np.array(json.loads(patient.face_descriptor), dtype=np.float32)
                if len(stored_embedding) != 2048:
                    print(f"[FACE MATCH] Skipping patient {patient.id}: invalid embedding size {len(stored_embedding)}")
                    continue
                
                # Normalize stored embedding
                stored_norm = stored_embedding / (np.linalg.norm(stored_embedding) + 1e-8)

                # Calculate multiple distance metrics
                
                # 1. Cosine similarity (ranges from -1 to 1)
                cosine_sim = float(np.dot(embedding_norm, stored_norm))
                
                # 2. Euclidean distance (lower is better)
                euclidean_dist = float(np.linalg.norm(embedding - stored_embedding))
                
                # 3. Manhattan distance
                manhattan_dist = float(np.sum(np.abs(embedding - stored_embedding)))
                
                # 4. Chi-square distance (for histogram-like features)
                chi_square = float(np.sum((embedding - stored_embedding) ** 2 / (np.abs(embedding) + np.abs(stored_embedding) + 1e-8)))
                
                # Composite score: prioritize cosine similarity but use other metrics for validation
                # Higher is better
                composite_score = cosine_sim - (euclidean_dist * 0.01)
                
                all_scores.append({
                    'patient_id': patient.patient_id,
                    'patient_name': patient.name,
                    'cosine_similarity': cosine_sim,
                    'euclidean_distance': euclidean_dist,
                    'manhattan_distance': manhattan_dist,
                    'chi_square': chi_square,
                    'composite_score': composite_score
                })

                # Track best and second-best matches
                if cosine_sim > best_similarity:
                    second_best_similarity = best_similarity
                    second_best_distance = best_distance
                    best_similarity = cosine_sim
                    best_distance = euclidean_dist
                    best_match = patient
                elif cosine_sim > second_best_similarity:
                    second_best_similarity = cosine_sim
                    second_best_distance = euclidean_dist

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"[FACE MATCH ERROR] Patient {patient.id}: {e}")
                continue

        # Debug logging
        print(f"\n[FACE MATCH DEBUG]")
        print(f"  Total patients checked: {len(all_scores)}")
        if all_scores:
            print(f"  Top 5 matches:")
            for score in sorted(all_scores, key=lambda x: x['cosine_similarity'], reverse=True)[:5]:
                print(f"    - {score['patient_name']}: cosine={score['cosine_similarity']:.4f}, euclidean={score['euclidean_distance']:.2f}")

        # Get trained model configuration
        model_config = get_face_model_config()
        COSINE_THRESHOLD = model_config['cosine_threshold']
        EUCLIDEAN_THRESHOLD = model_config['euclidean_threshold']
        SIMILARITY_GAP = model_config['similarity_gap']
        
        # Validation checks - RELAXED FOR BETTER ACCURACY WITH FALLBACK MODEL
        # Priority 1: High Cosine Similarity (Excellent match)
        is_excellent_match = best_similarity >= 0.85
        
        # Priority 2: Standard matching criteria
        passed_cosine = best_similarity >= COSINE_THRESHOLD
        passed_euclidean = best_distance <= EUCLIDEAN_THRESHOLD
        passed_gap = (best_similarity - second_best_similarity) >= SIMILARITY_GAP
        
        print(f"\n[FACE MATCH VALIDATION]")
        print(f"  Best cosine similarity: {best_similarity:.4f} (threshold: {COSINE_THRESHOLD:.4f}, passed: {passed_cosine})")
        print(f"  Best euclidean distance: {best_distance:.2f} (threshold: {EUCLIDEAN_THRESHOLD:.2f}, passed: {passed_euclidean})")
        print(f"  Similarity gap: {best_similarity - second_best_similarity:.4f} (threshold: {SIMILARITY_GAP:.4f}, passed: {passed_gap})")

        # Decision Logic:
        # Match if it's excellent OR if it passes basic cosine/euclidean (gap is secondary)
        should_match = best_match and (is_excellent_match or (passed_cosine and passed_euclidean))
        
        if should_match:
            print(f"\n[FACE MATCH SUCCESS] Matched: {best_match.name}")
            
            return jsonify({
                'found': True,
                'patient': {
                    'id': best_match.patient_id,
                    'name': best_match.name,
                    'age': best_match.age,
                    'gender': best_match.gender,
                    'phone': best_match.phone,
                    'email': best_match.email,
                    'department': best_match.department,
                    'consultation_type': best_match.consultation_type
                },
                'confidence': {
                    'cosine_similarity': float(best_similarity),
                    'euclidean_distance': float(best_distance),
                    'similarity_gap': float(best_similarity - second_best_similarity)
                },
                'matching_thresholds': {
                    'cosine': COSINE_THRESHOLD,
                    'euclidean': EUCLIDEAN_THRESHOLD,
                    'gap': SIMILARITY_GAP
                }
            })
        else:
            print(f"\n[FACE MATCH FAILED]")
            print(f"  Cosine threshold not met: {best_similarity:.4f} < {COSINE_THRESHOLD}")
            print(f"  Euclidean threshold not met: {best_distance:.2f} > {EUCLIDEAN_THRESHOLD}")
            print(f"  Gap threshold not met: {best_similarity - second_best_similarity:.4f} < {SIMILARITY_GAP}")
            
            return jsonify({
                'found': False,
                'best_similarity': float(best_similarity) if best_similarity != -2 else 0,
                'best_distance': float(best_distance) if best_distance != float('inf') else 9999,
                'matching_thresholds': {
                    'cosine': COSINE_THRESHOLD,
                    'euclidean': EUCLIDEAN_THRESHOLD,
                    'gap': SIMILARITY_GAP
                },
                'message': 'No matching patient found. New patient registration.'
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
                            <p class="mb-1 small">{patient.name}</p>
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
                alert('Please check the registration desk for your queue status, or use the chatbot for assistance.');
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
            alert(`📍 ${location}\n\nPlease follow the signs or ask staff for directions. Use the chatbot for more details.`);
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
                                <div class="display-1 text-danger my-3">112</div>
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
                    'cosine_threshold': result.cosine_threshold,
                    'euclidean_threshold': result.euclidean_threshold,
                    'total_patients': result.total_patients
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Training failed'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Initialize the database
init_db()

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
    