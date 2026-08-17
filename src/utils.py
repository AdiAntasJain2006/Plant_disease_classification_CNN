"""
src/utils.py
Explainable AI (XAI) utility using Grad-CAM to generate visual heatmaps 
for EfficientNet predictions without relying on internal layer names.
"""

import cv2
import numpy as np
import tensorflow as tf


def get_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generates a Grad-CAM heatmap by directly watching the backbone's 4D feature map.
    This bypasses nested layer-name lookups completely.
    """
    # 1. Identify the backbone layer and the top classification head layers
    backbone = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        # Fallback to the second layer if naming differs
        backbone = model.layers[1]

    # Collect all layers following the backbone (Pooling, BN, Dropout, Dense)
    backbone_idx = model.layers.index(backbone)
    head_layers = model.layers[backbone_idx + 1:]

    # 2. Record operations using GradientTape on the backbone feature tensor
    with tf.GradientTape() as tape:
        # Get raw 4D feature map directly from base model (1, 7, 7, 1280)
        conv_outputs = backbone(img_array, training=False)
        tape.watch(conv_outputs)

        # Forward pass through the head layers
        x = conv_outputs
        for head_layer in head_layers:
            x = head_layer(x, training=False)
        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 3. Compute gradients of predicted class w.r.t the 4D conv_outputs feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # 4. Calculate channel importance weights & project heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs_val = conv_outputs[0]
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 5. Apply ReLU activation and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_heatmap(img_path, heatmap, alpha=0.4):
    """Overlays the Grad-CAM heatmap onto the original RGB image."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, color_heatmap, alpha, 0)
    return superimposed_img