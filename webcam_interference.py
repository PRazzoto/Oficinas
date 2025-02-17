#!/usr/bin/env python3
import cv2
import numpy as np
import time
import serial
import tensorflow as tf

# ----------------------------
# Configuration Parameters
# ----------------------------
MODEL_PATH = "fruit_classifier.tflite"  # TFLite model file
IMG_WIDTH = 150
IMG_HEIGHT = 150

SERIAL_PORT = "/dev/ttyACM0"  # Adjust if necessary (e.g., /dev/ttyUSB0)
BAUD_RATE = 9600

# ----------------------------
# Load TFLite Model
# ----------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mapping from predicted index to fruit name (alphabetical order assumed)
idx_to_class = {0: "apple", 1: "banana", 2: "orange"}

# ----------------------------
# Setup Serial Communication with Arduino
# ----------------------------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for the connection to stabilize
    print("Serial connection established on", SERIAL_PORT)
except Exception as e:
    print("Error opening serial port:", e)
    exit()

# ----------------------------
# Initialize Webcam
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    ser.close()
    exit()

print("Raspberry Pi ready. Waiting for trigger from Arduino...")

# ----------------------------
# Main Loop: Wait for Trigger from Arduino
# ----------------------------
while True:
    if ser.in_waiting > 0:
        # Read incoming serial data (strip newline)
        line = ser.readline().decode("utf-8").strip()
        print("Received from Arduino:", line)
        if line == "TRIGGER":
            print("Trigger received. Capturing image for classification...")
            # Open the webcam, capture a single frame, then release (to keep system light)
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not capture frame.")
                continue

            # Preprocess the captured frame
            resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            normalized_frame = rgb_frame.astype("float32") / 255.0
            input_array = np.expand_dims(normalized_frame, axis=0)

            # Run inference using the TFLite model
            interpreter.set_tensor(input_details[0]["index"], input_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]["index"])
            predicted_index = int(np.argmax(predictions, axis=1)[0])
            predicted_class = idx_to_class.get(predicted_index, "unknown")
            confidence = predictions[0][predicted_index]
            print(
                "Classification result:", predicted_class, "with confidence", confidence
            )

            # Send the classification result (as a number: 0, 1, or 2) back to the Arduino
            ser.write((str(predicted_index) + "\n").encode("utf-8"))
            print("Sent to Arduino:", predicted_index)

    # Small delay to avoid busy looping
    time.sleep(0.1)

# Cleanup (in case of exit)
cap.release()
ser.close()
