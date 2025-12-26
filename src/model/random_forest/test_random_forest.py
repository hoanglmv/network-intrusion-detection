
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define file paths
processed_data_dir = os.path.join(project_root, "data", "processed")
original_data_path = os.path.join(processed_data_dir, "data.csv")
test_data_path = os.path.join(processed_data_dir, "test_data.csv")
model_path = os.path.join(project_root, "trained", "random_forest_model.joblib")
report_dir = os.path.join(project_root, "report")
confusion_matrix_path = os.path.join(report_dir, "random_forest_confusion_matrix.png")

# Create report directory if it doesn't exist
os.makedirs(report_dir, exist_ok=True)

# Load the model and test data
model = joblib.load(model_path)
test_df = pd.read_csv(test_data_path)

# Load original data to fit LabelEncoder and get class names
original_df = pd.read_csv(original_data_path)
le = LabelEncoder()
le.fit(original_df['label'])
class_names = le.classes_

# Separate features and labels
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Note: Random Forest does not require feature scaling like some other models.
# We will use the data as is, consistent with the training script.

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names)

print(f"Random Forest Model Evaluation")
print("="*40)
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(report)

# Generate and save the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(confusion_matrix_path)
plt.close()

print(f"\nConfusion matrix saved to: {confusion_matrix_path}")
print("Test script execution finished.")
