# 🧠 MNIST Handwritten Digit Classification — Convolutional Neural Network

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.  
It demonstrates a complete deep-learning workflow, including data preprocessing, model architecture design, training, evaluation, and visualization.

<p align="center">
  <img src="recognition.gif" alt="Digit Recognition Demo" width="300">
</p>

---

## 📌 Project Overview

The goal of this project is to build a CNN capable of recognizing handwritten digits (0–9) with high accuracy.  
Using the MNIST dataset (70,000 grayscale images of size 28×28), the model learns to identify patterns in handwritten digits and predict the correct class.

This project showcases:
- 🧼 Data preprocessing for image-based deep learning  
- 🧩 Construction of a simple yet effective CNN  
- 📉 Training and validation using best practices  
- 📈 Evaluation on unseen test data  
- ⚙️ Understanding parameter counts and convolution operations  

---

## 🗂️ Repository Structure

.
├── mnist_classification.ipynb # Full training pipeline and analysis
├── recognition.gif # Demo animation
├── README.md # Project documentation
└── .gitignore

---

## 📊 Dataset

**MNIST Dataset**  
- 60,000 training images  
- 10,000 test images  
- Grayscale (1 channel), size 28×28  
- Classes: digits 0–9 (10-class classification)

Images are normalized to the 0–1 range and reshaped to `(28, 28, 1)` to match CNN input requirements.  
Labels are one-hot encoded for multiclass classification.

---

## 🧱 Model Architecture

The CNN consists of:

- **Conv2D (8 filters, 4×4, ReLU, padding="same")**  
- **MaxPooling2D (2×2)**  
- **Conv2D (16 filters, 4×4, ReLU)**  
- **MaxPooling2D (2×2)**  
- **Flatten**
- **Dense (10 units, ReLU)**
- **Dense (10 units, Softmax)** — classification output layer

**Compilation:**
- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Metric: `accuracy`

Total parameters: **6,320**

---

## 🚀 Training

The model is trained for **5 epochs** with:
- A `validation_split` of 0.3  
- Early stopping to preserve the best model  
- Batch size = 32  

On a typical CPU setup, training completes in under one minute.

---

## 📈 Results

**Validation Accuracy:** ~97%  
**Test Accuracy:** **98.17%**

```python
model.evaluate(X_test, y_test_cat)
# Output:
# [0.0586, 0.9817]
