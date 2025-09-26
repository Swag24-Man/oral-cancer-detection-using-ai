import os, json
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Print TensorFlow version and device info
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Built with AVX:", tf.pywrap_tensorflow.IsMklEnabled() if hasattr(tf, 'pywrap_tensorflow') else "Unknown")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

MODEL_PATH = os.path.join(BASE_DIR, "final_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 15

# ------------------------
# Data Augmentation
# ------------------------
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

# ------------------------
# Class Weights (fix imbalance)
# ------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# ------------------------
# Model
# ------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ------------------------
# Training
# ------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights
)

# ------------------------
# Save Model + Labels
# ------------------------
model.save(MODEL_PATH)

with open(LABELS_PATH, "w") as f:
    json.dump(train_gen.class_indices, f)

print(f"Model saved to {MODEL_PATH}")
print(f"Labels saved to {LABELS_PATH}")

# ------------------------
# Compile the loaded model (fix for metrics warning)
# ------------------------
loaded_model = tf.keras.models.load_model(MODEL_PATH)
loaded_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("Loaded model compiled successfully.")

