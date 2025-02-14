import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# ----------------------------
# Configuration and Model Loading
# ----------------------------
MODEL_PATH = "fruit_classifier.h5"  # Update if needed

# If you need the custom InputLayer workaround, uncomment and modify the next lines:
# class CustomInputLayer(tf.keras.layers.InputLayer):
#     @classmethod
#     def from_config(cls, config):
#         if "batch_shape" in config:
#             config["batch_input_shape"] = config.pop("batch_shape")
#         return super(CustomInputLayer, cls).from_config(config)
#
# model = load_model(MODEL_PATH, custom_objects={"InputLayer": CustomInputLayer})
model = load_model(MODEL_PATH)

# Map predicted indices to fruit names (assumes alphabetical ordering: apple, banana, orange)
idx_to_class = {0: "apple", 1: "banana", 2: "orange"}

# Expected input size for the model
IMG_WIDTH = 150
IMG_HEIGHT = 150

# ----------------------------
# Initialize Webcam and Background Subtractor
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Use a background subtractor to detect motion
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=True
)
motion_threshold = 5000  # Adjust based on your environment

# Flags for detection
detection_mode = False
detection_start_time = None

print("System armed. Waiting for an object to enter the frame... (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to detect motion
    fg_mask = bg_subtractor.apply(frame)
    motion_amount = np.sum(fg_mask) / 255  # counts white pixels in the mask

    # If an object enters the frame (motion detected) and we're not already in detection mode:
    if not detection_mode and motion_amount > motion_threshold:
        detection_mode = True
        detection_start_time = time.time()
        print("Object detected. Waiting 3 seconds before classification...")

    # Once in detection mode, wait until 3 seconds have passed.
    if detection_mode and (time.time() - detection_start_time) >= 3:
        # Capture the frame for classification
        resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = rgb_frame.astype("float32") / 255.0
        input_array = np.expand_dims(normalized_frame, axis=0)

        # Perform classification
        predictions = model.predict(input_array)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        predicted_class = idx_to_class.get(predicted_index, "Unknown")
        confidence = predictions[0][predicted_index]
        result_text = f"{predicted_class}: {confidence*100:.2f}%"
        print("Classification result:", result_text)

        # Overlay the result on the frame and display it for 3 seconds
        cv2.putText(
            frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Result", frame)
        cv2.waitKey(3000)  # display result for 3 seconds

        # After classification, wait for the object to leave the frame before re-arming.
        print("Waiting for object to leave the frame...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fg_mask = bg_subtractor.apply(frame)
            motion_amount = np.sum(fg_mask) / 255
            cv2.imshow("Webcam", frame)
            if motion_amount < motion_threshold:
                detection_mode = False  # re-arm detection
                print("Frame is clear. System re-armed.")
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                detection_mode = False
                break

    # Show the live webcam feed and the motion mask for debugging (optional)
    cv2.imshow("Webcam", frame)
    cv2.imshow("Motion Mask", fg_mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
