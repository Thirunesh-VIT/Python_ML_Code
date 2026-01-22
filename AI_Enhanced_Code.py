import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 1) Load and split data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

# 2) Scaling (Crucial for convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3) Define a search space for Hyperparameter Tuning
# We test different layer sizes and learning rates
param_grid = {
    'hidden_layer_sizes': [(64, 32), (100,), (50, 50)],
    'alpha': [0.0001, 0.05],  # L2 penalty (regularization)
    'learning_rate_init': [0.001, 0.01],
}

# 4) Initialize MLP with Early Stopping
# early_stopping=True sets aside 10% of training data as validation to prevent overfitting
mlp = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)

# 5) Use GridSearchCV to find the best model
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_mlp = grid_search.best_estimator_

# 6) Evaluation
print(f"Best Parameters: {grid_search.best_params_}")
y_pred = best_mlp.predict(X_test_scaled)

# 7) Visualizing Results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot Learning Curve (Loss over time)
ax[0].plot(best_mlp.loss_curve_)
ax[0].set_title("Loss Curve (Training Convergence)")
ax[0].set_xlabel("Iterations")
ax[0].set_ylabel("Loss")

# Plot Confusion Matrix
ConfusionMatrixDisplay.from_estimator(best_mlp, X_test_scaled, y_test, 
                                      display_labels=data.target_names, 
                                      cmap='Blues', ax=ax[1])
ax[1].set_title("Confusion Matrix")

plt.tight_layout()
plt.show()

print("\nFinal Classification Report:\n", classification_report(y_test, y_pred))
