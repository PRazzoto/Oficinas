# webcam_inference.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ----------------------------
# Load the Trained Model
# ----------------------------
model = load_model("fruit_classifier.h5")

# ----------------------------
# Define Class Mapping
# ----------------------------
# The training generator typically sorts classes alphabetically.
# Expected mapping: {'apple': 0, 'banana': 1, 'orange': 2}
idx_to_class = {0: "apple", 1: "banana", 2: "orange"}

IMG_HEIGHT = 150
IMG_WIDTH = 150

# ----------------------------
# Initialize Webcam
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. The classifier will update predictions continuously.")
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Preprocess the frame: resize, convert BGR to RGB, and normalize.
    resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    normalized_frame = rgb_frame.astype("float32") / 255.0
    input_array = np.expand_dims(normalized_frame, axis=0)

    # Make a prediction on the frame.
    predictions = model.predict(input_array)
    predicted_index = int(np.argmax(predictions, axis=1)[0])
    predicted_class = idx_to_class.get(predicted_index, "Unknown")
    confidence = predictions[0][predicted_index]
    output_text = f"{predicted_class}: {confidence*100:.2f}%"

    # Overlay the prediction text on the frame.
    cv2.putText(
        frame, output_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # Display the live video feed.
    cv2.imshow("Webcam", frame)

    # Exit the loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
