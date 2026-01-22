# Python_ML_Code
This README is designed to give your GitHub repository or project folder a professional look. It clearly explains the transition from a basic model to an AI-enhanced one.

---

# Breast Cancer Detection using Multi-Layer Perceptron (MLP)

This repository demonstrates the implementation of a Neural Network (Multi-Layer Perceptron) to classify breast cancer tumors as **Malignant** or **Benign** using the UCI Breast Cancer Wisconsin dataset.

The project showcases two stages of development:

1. **Baseline Model:** A standard Scikit-Learn implementation.
2. **AI-Enhanced Model:** An optimized version utilizing Hyperparameter Tuning, Early Stopping, and advanced visualization.

## 🚀 Features

* **Dataset:** Scikit-Learn's `load_breast_cancer` (30 features, 569 samples).
* **Preprocessing:** Robust feature scaling using `StandardScaler`.
* **Optimization:** Automated hyperparameter search with `GridSearchCV`.
* **Overfitting Protection:** Implementation of `early_stopping` to ensure model generalization.
* **Visualization:** Automated plotting of loss curves and confusion matrices.

## 📂 Project Structure

* `mlp_baseline.py`: The original, straightforward implementation.
* `mlp_enhanced.py`: The optimized version with grid search and visualization.
* `README.md`: Project documentation.

## 🛠️ Installation

Ensure you have Python 3.7+ installed. Clone this repository and install the required dependencies:

```bash
pip install numpy scikit-learn matplotlib

```

## 📈 Model Comparison

### Baseline Model

The initial model uses a fixed architecture of two hidden layers `(64, 32)` and achieves a high accuracy, but it lacks a way to verify if these settings are optimal for the data.

### Enhanced Model (AI-Assisted)

The enhanced version introduces:

* **Stratified Splitting:** Ensures class balance in training and testing sets.
* **Grid Search:** Tests multiple architectures (e.g., `(100,)`, `(50, 50)`) to find the best F1-score.
* **Convergence Monitoring:** Plots the **Loss Curve** to visualize how the model learns over time.
* **Visual Evaluation:** Uses `ConfusionMatrixDisplay` for a clearer look at False Positives vs. False Negatives.

## 📊 Results

The enhanced model typically achieves an accuracy/F1-score between **97% and 99%**. By using the AI-suggested `early_stopping` parameter, the model avoids training longer than necessary, saving computational resources and preventing overfitting.

## 🤖 AI Acknowledgement

The enhancements provided in the `mlp_enhanced.py` script, including the hyperparameter grid strategy and advanced visualization techniques, were developed with the assistance of **Gemini**, a large language model trained by Google. This collaboration highlights the power of using AI as a thought partner to move from functional code to optimized, production-ready machine learning pipelines.

---
