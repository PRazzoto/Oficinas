# train_fruit_classifier.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# ----------------------------
# Configuration Parameters
# ----------------------------
DATASET_DIR = "original_data_set"  # <-- UPDATE this to your dataset root folder
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
INITIAL_EPOCHS = 20  # Initial training with the base model frozen
FINE_TUNE_EPOCHS = 15  # Fine-tuning epochs
FINE_TUNE_AT = 50  # Unfreeze layers from this index onward (lower than before to allow more adaptation)

# ----------------------------
# Prepare the DataFrame
# ----------------------------
# Loop through each subfolder and assign a label based on folder name.
data = []
for subfolder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, subfolder)
    if os.path.isdir(folder_path):
        subfolder_lower = subfolder.lower()
        if "apple" in subfolder_lower:
            label = "apple"
        elif "banana" in subfolder_lower:
            label = "banana"
        elif "orange" in subfolder_lower:
            label = "orange"
        else:
            continue  # Skip folders that don't match our classes

        # Loop through image files in the folder
        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                file_path = os.path.join(folder_path, file)
                data.append([file_path, label])

# Create a DataFrame with columns "filename" and "class"
df = pd.DataFrame(data, columns=["filename", "class"])
print("Total images found:", len(df))
print("Class distribution:\n", df["class"].value_counts())

# ----------------------------
# Split Data into Training and Validation Sets
# ----------------------------
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["class"]
)

# ----------------------------
# Create Data Generators with Augmentation
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,  # Added shear to slightly distort images
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    horizontal_flip=True,
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filename",
    y_col="class",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filename",
    y_col="class",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# Print the class indices mapping (expected to be something like: {'apple': 0, 'banana': 1, 'orange': 2})
print("Class Indices:", train_generator.class_indices)

# ----------------------------
# Build the Transfer Learning Model (Initial Stage)
# ----------------------------
# Load MobileNetV2 (without top layers) pre-trained on ImageNet.
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # Freeze the base model initially

# Add custom classification layers on top.
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation="softmax")(x)  # Three classes: apple, banana, orange

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# ----------------------------
# Initial Training (Base Frozen)
# ----------------------------
history_initial = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
)

# ----------------------------
# Fine-Tuning: Unfreeze Some Base Model Layers
# ----------------------------
# Unfreeze layers from FINE_TUNE_AT onward so the model can adjust to your data.
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# Recompile the model with a lower learning rate.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
)

# ----------------------------
# (Optional) Analyze Performance with a Confusion Matrix
# ----------------------------
validation_generator.reset()
preds = model.predict(
    validation_generator, steps=validation_generator.samples // BATCH_SIZE + 1
)
y_pred = np.argmax(preds, axis=1)
y_true = validation_generator.classes

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=list(train_generator.class_indices.keys())
)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix on Validation Set")
plt.show()

# ----------------------------
# Save the Final Model
# ----------------------------
model.save("fruit_classifier.h5")
print("Model saved as 'fruit_classifier.h5'")
