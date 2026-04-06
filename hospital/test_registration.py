from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

print("Starting Automated Camera Registration Flow Test...")

options = Options()
# Force browser to use fake device so the camera feed works immediately without asking for permissions
options.add_argument("--use-fake-ui-for-media-stream")
options.add_argument("--use-fake-device-for-media-stream")
# optionally headless
# options.add_argument("--headless")

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

try:
    driver.get("http://localhost:5000/register")
    print("Navigated to register page.")

    # 1. Start Camera
    start_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "startCameraBtn"))
    )
    print("Clicking Start Camera...")
    start_btn.click()

    # 2. Wait for Capture Face to be enabled (meaning onloadedmetadata fired)
    capture_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "captureFaceBtn"))
    )
    print("Camera loaded successfully! Capture button is enabled via onloadedmetadata callback.")
    
    # 3. Click Capture Face
    print("Clicking Capture Face...")
    capture_btn.click()
    
    # 4. Wait for Face Capture Success message via faceStatus and step2Card
    step2_card = WebDriverWait(driver, 15).until(
        EC.visibility_of_element_located((By.ID, "step2Card"))
    )
    print("Step 2 Card became visible! This means the /api/register/detect-face backend call succeeded.")
    
    # Examine the DOM for the CaptureFace ID
    face_id_input = driver.find_element(By.ID, "capturedFaceIdInput")
    print(f"Captured Face ID saved: {face_id_input.get_attribute('value')}")

    time.sleep(2)
    print("Test Complete. Registration Flow Verification: PASSED")

except Exception as e:
    print(f"Test Failed! {e}")

finally:
    driver.quit()
