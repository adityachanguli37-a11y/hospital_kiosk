from hospital_kiosk_web import db, Patient, User, CapturedFace, EmergencyLog, ChatMessage, app

def clear_data():
    with app.app_context():
        print("[*] Starting Database Cleanup...")
        
        # 1. Delete Emergency Logs (linked to patients)
        count_emergency = EmergencyLog.query.delete()
        print(f"[*] Deleted {count_emergency} Emergency Logs.")
        
        # 2. Delete Patients
        count_patients = Patient.query.delete()
        print(f"[*] Deleted {count_patients} Patient records.")
        
        # 3. Delete User accounts for patients
        count_users = User.query.filter_by(role='patient').delete()
        print(f"[*] Deleted {count_users} Patient user accounts.")
        
        # 4. Delete Captured Faces
        count_faces = CapturedFace.query.delete()
        print(f"[*] Deleted {count_faces} Captured Face records.")
        
        # 5. Delete Chat Messages (Optional, for a clean start)
        count_chats = ChatMessage.query.delete()
        print(f"[*] Deleted {count_chats} Chat Messages.")
        
        db.session.commit()
        print("\n[OK] Cleanup Complete! All patient data has been removed.")
        print("[NOTE] Doctor accounts and Department data remain intact.")

if __name__ == "__main__":
    clear_data()
