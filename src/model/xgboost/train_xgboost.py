
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define file paths
processed_data_dir = "D:\\vhproj\\network-intrusion-detection\\data\\processed"
train_data_path = os.path.join(processed_data_dir, "train_data.csv")
test_data_path = os.path.join(processed_data_dir, "test_data.csv")
val_data_path = os.path.join(processed_data_dir, "val_data.csv")
model_output_path = "D:\\vhproj\\network-intrusion-detection\\trained\\xgboost_model.joblib"

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

# Initialize and train the XGBoost model
# For multi-class classification, use 'multi:softmax' or 'multi:softprob'
# And set num_class to the number of unique classes
num_classes = len(y_train.unique())
model = XGBClassifier(objective='multi:softmax', num_class=num_classes, n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss')
eval_set = [(X_val, y_val)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

# Save training history
history = model.evals_result()
history_df = pd.DataFrame(history['validation_0'])
history_df.to_csv('D:\\vhproj\\network-intrusion-detection\\trained\\xgboost_history.csv', index=False)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"XGBoost Model Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Save the trained model
joblib.dump(model, model_output_path)

print(f"Model saved to {model_output_path}")
