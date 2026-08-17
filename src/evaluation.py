"""
src/evaluation.py
Evaluates trained model performance on validation data and plots metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def evaluate_model(model_path, val_ds, class_names):
    """
    Computes precision, recall, f1-score, and draws a confusion matrix.
    """
    # Load trained model
    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []

    # Extract ground truth and predictions
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    # Print Text Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Plant Disease Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png")
    print("[INFO] Confusion matrix saved to assets/confusion_matrix.png")

if __name__ == "__main__":
    print("Evaluation module ready.")