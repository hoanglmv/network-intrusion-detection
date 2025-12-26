import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt # Thư viện vẽ đồ thị

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define file paths
processed_data_dir = os.path.join(project_root, "data", "processed")
original_data_path = os.path.join(processed_data_dir, "data.csv")
test_data_path = os.path.join(processed_data_dir, "test_data.csv")
model_path = os.path.join(project_root, "trained", "cnn_model.h5")

# Paths for history and reports
history_path = os.path.join(project_root, "trained", "cnn_history.csv")
report_dir = os.path.join(project_root, "report")
confusion_matrix_path = os.path.join(report_dir, "cnn_confusion_matrix.png")
loss_plot_report_path = os.path.join(report_dir, "cnn_training_loss.png") # Đường dẫn ảnh biểu đồ trong report

# Create report directory if it doesn't exist
os.makedirs(report_dir, exist_ok=True)

# Load the model and test data
model = tf.keras.models.load_model(model_path)
test_df = pd.read_csv(test_data_path)

# Load original data to fit LabelEncoder and get class names
original_df = pd.read_csv(original_data_path)
le = LabelEncoder()
le.fit(original_df['label'])
class_names = le.classes_

# Separate features and labels
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Preprocess the data using the same scaler and reshaping from training
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
X_test_reshaped = np.expand_dims(X_test_scaled, axis=2)

# Make predictions
pred_probs = model.predict(X_test_reshaped)
y_pred = np.argmax(pred_probs, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names)

print(f"CNN Model Evaluation")
print("="*40)
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(report)

# Generate and save the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('CNN Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(confusion_matrix_path)
plt.close()
print(f"\nConfusion matrix saved to: {confusion_matrix_path}")

# --- VẼ LẠI BIỂU ĐỒ LOSS TỪ FILE HISTORY CHO BÁO CÁO ---
if os.path.exists(history_path):
    print("Generating loss plot from training history...")
    history_df = pd.read_csv(history_path)
    
    plt.figure(figsize=(12, 5))

    # Vẽ biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_df['loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # Vẽ biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history_df['accuracy'], label='Train Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(loss_plot_report_path)
    plt.close()
    print(f"Loss/Accuracy plot saved to: {loss_plot_report_path}")
else:
    print(f"Warning: Could not find history file at {history_path}. Skipping plot generation.")

print("Test script execution finished.")