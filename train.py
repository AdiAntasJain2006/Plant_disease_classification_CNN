"""
train.py
Main orchestration script to load datasets, compile EfficientNetB0,
and run two-stage transfer learning using sparse categorical crossentropy.
"""

import os
import tensorflow as tf
from src.dataset import load_datasets
from src.model import build_model, unfreeze_model

# Hyperparameters & Paths
DATA_DIR = "data"
MODEL_SAVE_PATH = "models/plant_model.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 10
NUM_CLASSES = 38


def train():
    os.makedirs("models", exist_ok=True)

    # 1. Load Data Streams
    print("[INFO] Loading datasets...")
    train_ds, val_ds, class_names = load_datasets(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # 2. Build EfficientNetB0 Architecture
    print("[INFO] Building EfficientNetB0 model...")
    model, base_model = build_model(num_classes=NUM_CLASSES, input_shape=(*IMG_SIZE, 3))

    # 3. Keras Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 4. Stage 1: Feature Extraction (Frozen Base)
    print("\n[INFO] Starting Stage 1 Training (Feature Extraction)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        callbacks=callbacks
    )

    # 5. Stage 2: Fine-Tuning (Unfreeze Top 20 Layers)
    print("\n[INFO] Starting Stage 2 Training (Fine-Tuning)...")
    unfreeze_model(base_model, num_layers_to_unfreeze=20)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE2_EPOCHS,
        callbacks=callbacks
    )

    print(f"\n[INFO] Training complete! Best model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()