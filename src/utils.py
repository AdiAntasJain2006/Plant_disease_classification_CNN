"""
src/utils.py
Explainable AI (XAI) utility using Grad-CAM to generate visual heatmaps 
for EfficientNet predictions.
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv", pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given input image and model prediction.

    Args:
        img_array (np.ndarray): Preprocessed image tensor of shape (1, 224, 224, 3).
        model (tf.keras.Model): Trained EfficientNet Keras model.
        last_conv_layer_name (str): Name of the target convolutional layer.
        pred_index (int, optional): Target class index. Uses top prediction if None.

    Returns:
        np.ndarray: Normalized 2D Grad-CAM heatmap array.
    """
    # 1. Build a sub-model mapping inputs to top conv layer and final output
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 3. Compute gradients of predicted class w.r.t. convolutional feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # 4. Compute mean intensity of gradients per feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply feature map outputs by gradient importance weights
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU (keep positive influences) and normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(img_path, heatmap, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap onto the original RGB image.

    Args:
        img_path (str): File path to original image.
        heatmap (np.ndarray): 2D heatmap matrix output from get_gradcam_heatmap().
        alpha (float): Opacity blend factor for heatmap overlay.

    Returns:
        np.ndarray: Superimposed RGB image ready for display.
    """
    # Load and resize original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to 8-bit uint8 and apply JET colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, color_heatmap, alpha, 0)
    return superimposed_img


if __name__ == "__main__":
    print("Grad-CAM utility ready for inference!")