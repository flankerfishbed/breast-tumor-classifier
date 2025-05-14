import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import io

# Load model
model = load_model("breast_tumor_classifier.h5")

# Grad-CAM heatmap generator
def generate_gradcam(image):
    image_resized = image.resize((224, 224)).convert("RGB")
    img_array = img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer("Conv_1").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Overlay on image
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image_np = np.array(image_resized)
    overlay = cv2.addWeighted(image_np, 0.5, heatmap_color, 0.5, 0)

    return Image.fromarray(overlay)

# Classifier + Grad-CAM function
def classify_and_visualize(image):
    img = image.resize((224, 224)).convert("RGB")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "Malignant" if pred > 0.5 else "Benign"
    confidence = f"{pred:.2f}" if pred > 0.5 else f"{1 - pred:.2f}"
    gradcam_image = generate_gradcam(image)

    return f"{label} (Confidence: {confidence})", gradcam_image

# Gradio interface
demo = gr.Interface(
    fn=classify_and_visualize,
    inputs=gr.Image(type="pil", label="Upload Breast Ultrasound"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Image(type="pil", label="Grad-CAM Heatmap")
    ],
    title="Breast Tumor Classifier (Ultrasound) with Grad-CAM",
    description="Upload a breast ultrasound image. The model will classify the tumor and show where it focused."
)

demo.launch()

