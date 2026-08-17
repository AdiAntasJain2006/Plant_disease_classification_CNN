"""
src/model.py
Defines the EfficientNetB0 Transfer Learning architecture 
and two-stage fine-tuning helper routines.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(num_classes=38, input_shape=(224, 224, 3)):
    """
    Builds the model with a frozen EfficientNetB0 backbone (Stage 1).
    """
    # 1. Load pre-trained base model
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Freeze base weights for initial head training
    base_model.trainable = False

    # 2. Add classification head
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pooling")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, name="top_dropout")(x)
    
    # Softmax probabilities for 38 disease categories
    outputs = layers.Dense(num_classes, activation="softmax", name="pred_layer")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Plant_Disease")
    return model, base_model


def unfreeze_model(base_model, num_layers_to_unfreeze=20):
    """
    Unfreezes the top N layers for Stage 2 fine-tuning.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    print(f"[INFO] Unfroze top {num_layers_to_unfreeze} layers for fine-tuning.")


if __name__ == "__main__":
    model, base = build_model(num_classes=38)
    model.summary()