
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define file paths
processed_data_dir = "D:\\vhproj\\network-intrusion-detection\\data\\processed"
train_data_path = os.path.join(processed_data_dir, "train_data.csv")
test_data_path = os.path.join(processed_data_dir, "test_data.csv")
model_output_path = "D:\\vhproj\\network-intrusion-detection\\trained\\logistic_regression_model.joblib"

# Load the datasets
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

from sklearn.preprocessing import StandardScaler

# Separate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Linear Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Linear Regression Model Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Save the trained model
joblib.dump(model, model_output_path)

print(f"Model saved to {model_output_path}")
