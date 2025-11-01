
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define file paths
processed_data_dir = "D:\\vhproj\\intrusion-network\\data\\processed"
train_data_path = os.path.join(processed_data_dir, "train_data.csv")
test_data_path = os.path.join(processed_data_dir, "test_data.csv")
model_output_path = "D:\\vhproj\\intrusion-network\\trained\\linear_regression_model.joblib"

# Load the datasets
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Separate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_float = model.predict(X_test)
y_pred = y_pred_float.round().astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Linear Regression Model Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Save the trained model
joblib.dump(model, model_output_path)

print(f"Model saved to {model_output_path}")
