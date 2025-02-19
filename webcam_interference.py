#!/usr/bin/env python3
import cv2
import numpy as np
import time
import serial
import tensorflow as tf
import os

# ----------------------------
# Configuration Parameters
# ----------------------------
MODEL_PATH = "fruit_classifier.tflite"  # Ensure this file is in the same directory
IMG_WIDTH = 150
IMG_HEIGHT = 150
SERIAL_PORT = "/dev/ttyACM0"  # On Windows; on Linux, use /dev/ttyACM0 or /dev/ttyUSB0
BAUD_RATE = 115200
delay_after_trigger = 2  # Delay in seconds after a classification is done

# Directory to save evaluated images
SAVE_DIR = "evaluated_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ----------------------------
# Load TFLite Model
# ----------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mapping from predicted index to fruit name (assumes alphabetical order: apple, banana, orange)
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

print("Computer ready. Waiting for trigger from Arduino...")

# ----------------------------
# Main Loop: Wait for Trigger from Arduino and Process Frame
# ----------------------------
while True:
    if ser.in_waiting > 0:
        # Read incoming serial data (strip newline)
        line = ser.readline().decode("utf-8").strip()
        print("Received from Arduino:", line)
        if line == "TRIGGER":
            print("Trigger received. Capturing image for classification...")

            # Open the webcam only when triggered, capture a frame, then release it
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                continue
            ret, frame = cap.read()
            cap.release()  # Release webcam immediately
            if not ret:
                print("Error: Could not capture frame.")
                continue

            # Save the captured frame with a timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(SAVE_DIR, f"image_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)
            print("Saved evaluated image to:", save_path)

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
            result_to_send = "F" + str(predicted_index + 1) + "\n"
            ser.write(result_to_send.encode("utf-8"))
            print("Sent to Arduino:", result_to_send.strip())

            # Wait for a specified delay before processing the next trigger
            print(f"Waiting {delay_after_trigger} seconds before next capture...")
            time.sleep(delay_after_trigger)

    # Small delay to avoid busy looping
    time.sleep(0.1)

# Cleanup (if loop ever exits)
ser.close()
