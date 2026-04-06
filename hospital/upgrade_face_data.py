from hospital_kiosk_web import db, Patient, FaceRecognitionModel, app
from PIL import Image
import io
import base64
import numpy as np
import json
import cv2
from scipy import ndimage

def get_new_embedding(image_data):
    """
    UPGRADED: Aligned Grid LBP logic
    """
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image_pil = Image.open(io.BytesIO(image_bytes))
    image_rgb = np.array(image_pil.convert('RGB'))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # 1. Alignment
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 10)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            p1 = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            p2 = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            angle = np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))
            center = (float((p1[0]+p2[0])//2 + x), float((p1[1]+p2[1])//2 + y))
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image_bgr, M, (image_bgr.shape[1], image_bgr.shape[0]))
            image_bgr = rotated[y:y+h, x:x+w].copy()
        else:
            image_bgr = image_bgr[y:y+h, x:x+w].copy()

    # 2. Features
    face_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).resize((128, 128)).convert('L')
    pixels = np.array(face_img).astype(np.float32) / 255.0
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    pixels = clahe.apply((pixels * 255).astype(np.uint8)).astype(np.float32) / 255.0

    features = []
    # Global (32x32)
    low_res = np.array(Image.fromarray((pixels*255).astype(np.uint8)).resize((32,32))).flatten()/255.0
    features.extend(low_res)
    
    # 8x8 Grid
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
            features.extend([np.mean(region), np.std(region), np.max(region), np.min(region)])
            features.extend(get_lbp_stats(region))

    # Gradients
    gx = ndimage.sobel(pixels, axis=1)
    gy = ndimage.sobel(pixels, axis=0)
    gx_small = np.array(Image.fromarray(gx).resize((16, 16))).flatten()
    gy_small = np.array(Image.fromarray(gy).resize((16, 16))).flatten()
    features.extend(gx_small / (np.max(np.abs(gx_small)) + 1e-8))
    features.extend(gy_small / (np.max(np.abs(gy_small)) + 1e-8))
    
    # Hist
    hist, _ = np.histogram(pixels.flatten(), bins=32, range=(0,1))
    features.extend(hist / (np.sum(hist)+1e-8))

    embedding_arr = np.array(features, dtype=np.float32)
    if len(embedding_arr) < 2048:
        embedding_arr = np.pad(embedding_arr, (0, 2048-len(embedding_arr)), mode='constant')
    else:
        embedding_arr = embedding_arr[:2048]
    
    norm = np.linalg.norm(embedding_arr)
    if norm > 0: embedding_arr /= norm
    return embedding_arr.tolist()

def upgrade():
    with app.app_context():
        print("[*] Starting HIGH-ACCURACY Face Data Migration...")
        model = FaceRecognitionModel.query.first()
        model.cosine_threshold = 0.85 # Even tighter for aligned model
        
        patients = Patient.query.filter(Patient.face_image.isnot(None)).all()
        count = 0
        for p in patients:
            print(f"[*] Aligned Upgrade: {p.name}...")
            try:
                new_embedding = get_new_embedding(p.face_image)
                p.face_descriptor = json.dumps(new_embedding)
                count += 1
            except Exception as e:
                print(f"    [X] Failed: {e}")
        db.session.commit()
        print(f"\n[*] Migration Complete! {count} patients upgraded to AlignedGridLBP.")

if __name__ == "__main__":
    upgrade()
