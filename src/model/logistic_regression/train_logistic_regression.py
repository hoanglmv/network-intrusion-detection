import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np
import warnings

# Bỏ qua cảnh báo ConvergenceWarning vì chúng ta chủ động lặp từng bước
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define file paths
processed_data_dir = os.path.join(project_root, "data", "processed")
train_data_path = os.path.join(processed_data_dir, "train_data.csv")
test_data_path = os.path.join(processed_data_dir, "test_data.csv")
model_output_path = os.path.join(project_root, "trained", "logistic_regression_model.joblib")
loss_plot_output_path = os.path.join(project_root, "trained", "logistic_regression_loss.png") # Đường dẫn lưu ảnh loss
acc_plot_output_path = os.path.join(project_root, "trained", "logistic_regression_accuracy.png") # Đường dẫn lưu ảnh accuracy

# Load the datasets
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Separate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- CẤU HÌNH TRAIN ĐỂ VẼ LOSS VÀ ACCURACY ---

# Số lượng epoch muốn chạy
epochs = 50 
train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

# Khởi tạo model với warm_start=True
model = LogisticRegression(warm_start=True, max_iter=1, solver='lbfgs')

print(f"Training Logistic Regression for {epochs} epochs...")

# Vòng lặp training thủ công
for epoch in range(epochs):
    # Train 1 bước
    model.fit(X_train_scaled, y_train)
    
    # --- Tính Loss ---
    y_train_prob = model.predict_proba(X_train_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)
    
    loss_train = log_loss(y_train, y_train_prob)
    loss_test = log_loss(y_test, y_test_prob)
    
    train_loss_history.append(loss_train)
    test_loss_history.append(loss_test)
    
    # --- Tính Accuracy ---
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    
    train_acc_history.append(acc_train)
    test_acc_history.append(acc_test)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss_train:.4f} - Test Loss: {loss_test:.4f} | Train Acc: {acc_train:.4f} - Test Acc: {acc_test:.4f}")

# --- VẼ VÀ LƯU ĐỒ THỊ LOSS ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_loss_history, label='Train Loss')
plt.plot(range(1, epochs + 1), test_loss_history, label='Test Loss', linestyle='--')
plt.title('Logistic Regression Training Loss over Epochs')
plt.xlabel('Epoch (Iterations)')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.savefig(loss_plot_output_path)
plt.close()
print(f"Loss plot saved to {loss_plot_output_path}")

# --- VẼ VÀ LƯU ĐỒ THỊ ACCURACY ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_acc_history, label='Train Accuracy')
plt.plot(range(1, epochs + 1), test_acc_history, label='Test Accuracy', linestyle='--')
plt.title('Logistic Regression Training Accuracy over Epochs')
plt.xlabel('Epoch (Iterations)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(acc_plot_output_path)
plt.close()
print(f"Accuracy plot saved to {acc_plot_output_path}")


# --- ĐÁNH GIÁ CUỐI CÙNG ---

# Evaluate the model on the final predictions
accuracy = accuracy_score(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred)

print(f"\nFinal Logistic Regression Model Accuracy: {accuracy}")
print("Final Classification Report:")
print(report)

# Save the trained model
joblib.dump(model, model_output_path)
print(f"Model saved to {model_output_path}")