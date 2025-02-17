#!/usr/bin/env python3
import cv2
import numpy as np
import time
import tensorflow as tf

# ----------------------------
# Configuration and TFLite Model Loading
# ----------------------------
MODEL_PATH = "fruit_classifier.tflite"  # Ensure this file is in the same directory
IMG_WIDTH = 150
IMG_HEIGHT = 150

# Load the TFLite model using TensorFlow's built-in TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Retrieve input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Map predicted indices to fruit names (assumes alphabetical ordering: apple, banana, orange)
idx_to_class = {0: "apple", 1: "banana", 2: "orange"}


def capture_and_classify():
    # Open the webcam only when triggered
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not capture an image.")
        return

    # Pre-process the frame
    resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    normalized_frame = rgb_frame.astype("float32") / 255.0
    input_array = np.expand_dims(normalized_frame, axis=0)

    # Run inference with the TFLite model
    interpreter.set_tensor(input_details[0]["index"], input_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])

    predicted_index = int(np.argmax(predictions, axis=1)[0])
    predicted_class = idx_to_class.get(predicted_index, "unknown")
    confidence = predictions[0][predicted_index]
    result_text = f"{predicted_class}: {confidence*100:.2f}%"
    print("Classification result:", result_text)

    # Optionally, display the captured image with the classification result overlayed
    cv2.putText(
        frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.imshow("Result", frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def main():
    print("System ready. Type 't' to trigger classification, or 'q' to quit.")
    while True:
        user_input = input("Enter command (t = trigger, q = quit): ").strip().lower()
        if user_input == "q":
            print("Exiting...")
            break
        elif user_input == "t":
            print("Trigger received. Capturing image...")
            capture_and_classify()
        else:
            print("Invalid command. Please type 't' or 'q'.")


if __name__ == "__main__":
    main()
