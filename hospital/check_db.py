from hospital_kiosk_web import db, Patient, CapturedFace, app
import json

with app.app_context():
    print(f"Total Patients: {Patient.query.count()}")
    print(f"Total Captured Faces: {CapturedFace.query.count()}")
    
    patients = Patient.query.filter(Patient.face_descriptor.isnot(None)).all()
    print(f"Patients with descriptors: {len(patients)}")
    for p in patients:
        try:
            desc = json.loads(p.face_descriptor)
            print(f"  Patient: {p.name}, Descriptor size: {len(desc)}")
        except Exception as e:
            print(f"  Patient: {p.name}, Error: {e}")
            
    from hospital_kiosk_web import FaceRecognitionModel
    model = FaceRecognitionModel.query.first()
    if model:
        print(f"\nModel Configuration:")
        print(f"  Cosine Threshold: {model.cosine_threshold}")
        print(f"  Euclidean Threshold: {model.euclidean_threshold}")
        print(f"  Similarity Gap: {model.similarity_gap}")
