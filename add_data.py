import cv2
import os
import time

# --- Configuration ---
# Directory setup: Images will be saved to original_data_set/freshapples
base_folder = "original_data_set"
label = "freshoranges"
class_dir = os.path.join(base_folder, label)
os.makedirs(class_dir, exist_ok=True)

# Image settings
IMG_WIDTH = 150
IMG_HEIGHT = 150

# Capture settings
capture_interval = 0.3  # seconds between captures
max_images = 2000  # Maximum number of images to capture (set as needed)

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting automatic image capture.")
print("Press 'q' in the display window to stop early.")

img_counter = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam - Automatic Capture", frame)

    current_time = time.time()
    if current_time - start_time >= capture_interval:
        resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        filename = os.path.join(class_dir, f"{label}_{img_counter:04d}.jpg")
        cv2.imwrite(filename, resized_frame)
        print(f"Saved {filename}")
        img_counter += 1
        start_time = current_time

    if cv2.waitKey(1) & 0xFF == ord("q") or img_counter >= max_images:
        print("Stopping capture.")
        break

cap.release()
cv2.destroyAllWindows()
