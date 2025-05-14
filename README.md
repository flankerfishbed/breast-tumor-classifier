---
title: Breast Tumor Classifier (Ultrasound)
emoji: 🧠
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# 🧠 Breast Tumor Classifier (Ultrasound)

A deep learning model trained to classify breast ultrasound tumor images as benign or malignant. Built with TensorFlow, fine-tuned on the BUS_UC dataset, and deployed using Gradio + Hugging Face Spaces.

## 💻 How to Use
1. Upload an ultrasound image.
2. The model predicts the tumor type:
   - Benign
   - Malignant
3. Shows a confidence score with each result.

## 📊 Model Details
- **Architecture**: MobileNetV2 (transfer learning)
- **Input**: 224x224 RGB ultrasound image
- **Output**: Binary classification
- **Accuracy**: ~69% on validation set

## 🛠 Built With
- TensorFlow
- Gradio
- PIL & NumPy
- Hugging Face Spaces

---

Built by [@flankerfish](https://huggingface.co/flankerfish)
