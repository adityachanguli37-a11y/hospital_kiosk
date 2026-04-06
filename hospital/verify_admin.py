from hospital_kiosk_web import app, db, User
import os

def verify():
    # Force absolute path to the correct database in instance folder
    db_path = os.path.join(app.root_path, 'instance', 'hospital_kiosk.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    
    with app.app_context():
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print("[!] Admin user NOT found! Creating now...")
            admin = User(username='admin', full_name='System Admin', role='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("[OK] Admin user 'admin' created with password 'admin123'")
        else:
            print(f"[OK] Admin user found: {admin.username} (Role: {admin.role})")
            # Force reset password just in case
            admin.set_password('admin123')
            db.session.commit()
            print("[OK] Password reset to 'admin123' to ensure login works.")

if __name__ == "__main__":
    verify()
