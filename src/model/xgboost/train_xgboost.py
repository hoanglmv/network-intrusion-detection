
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define file paths
processed_data_dir = os.path.join(project_root, "data", "processed")
train_data_path = os.path.join(processed_data_dir, "train_data.csv")
test_data_path = os.path.join(processed_data_dir, "test_data.csv")
val_data_path = os.path.join(processed_data_dir, "val_data.csv")
model_output_path = os.path.join(project_root, "trained", "xgboost_model.joblib")
loss_plot_output_path = os.path.join(project_root, "trained", "xgboost_loss.png")
acc_plot_output_path = os.path.join(project_root, "trained", "xgboost_accuracy.png")

# Load the datasets
train_df = pd.read_csv(train_data_path)
val_df = pd.read_csv(val_data_path)
test_df = pd.read_csv(test_data_path)

# Separate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_val = val_df.drop('label', axis=1)
y_val = val_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Initialize the XGBoost model
num_classes = len(y_train.unique())
model = XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    eval_metric=['mlogloss', 'merror']  # Track both logloss and multiclass error rate
)

# Define the evaluation set to monitor training and validation performance
eval_set = [(X_train, y_train), (X_val, y_val)]

print("Training XGBoost model...")
# Train the model and store the history
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
print("Training complete.")

# --- PLOT AND SAVE TRAINING HISTORY ---
history = model.evals_result()

# Extract loss and accuracy data
epochs = len(history['validation_0']['mlogloss'])
x_axis = range(0, epochs)

# Loss Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_axis, history['validation_0']['mlogloss'], label='Train Logloss')
ax.plot(x_axis, history['validation_1']['mlogloss'], label='Validation Logloss')
ax.legend()
plt.ylabel('Logloss')
plt.xlabel('Boosting Round')
plt.title('XGBoost Logloss')
plt.grid(True)
plt.savefig(loss_plot_output_path)
plt.close(fig)
print(f"Loss plot saved to {loss_plot_output_path}")

# Accuracy Plot (calculated from merror)
train_merror = history['validation_0']['merror']
val_merror = history['validation_1']['merror']
train_acc = [1 - x for x in train_merror]
val_acc = [1 - x for x in val_merror]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_axis, train_acc, label='Train Accuracy')
ax.plot(x_axis, val_acc, label='Validation Accuracy')
ax.legend()
plt.ylabel('Accuracy')
plt.xlabel('Boosting Round')
plt.title('XGBoost Accuracy')
plt.grid(True)
plt.savefig(acc_plot_output_path)
plt.close(fig)
print(f"Accuracy plot saved to {acc_plot_output_path}")


# --- FINAL EVALUATION ---
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nFinal XGBoost Model Accuracy: {accuracy}")
print("Final Classification Report:")
print(report)

# Save the trained model
joblib.dump(model, model_output_path)
print(f"Model saved to {model_output_path}")
