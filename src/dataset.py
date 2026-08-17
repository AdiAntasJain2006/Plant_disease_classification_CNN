"""
src/dataset.py
Production data pipeline using integer labels for sparse categorical crossentropy.
"""

import tensorflow as tf
from tensorflow.keras import layers

def get_data_augmentation():
    """Applies spatial transformations to raw image tensors during training."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

def load_datasets(data_dir, img_size=(224, 224), batch_size=32, seed=42):
    train_dir = f"{data_dir}/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
    val_dir = f"{data_dir}/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

    # Integer labels required for sparse_categorical_crossentropy
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        seed=seed
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        seed=seed
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names