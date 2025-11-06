
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define file paths
processed_data_dir = os.path.join(project_root, "data", "processed")
test_data_path = os.path.join(processed_data_dir, "test_data.csv")
model_path = os.path.join(project_root, "trained", "cnn_model.h5")

# Load the model and test data
model = tf.keras.models.load_model(model_path)
test_df = pd.read_csv(test_data_path)

# Separate features and labels
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Preprocess the data
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
X_test_reshaped = np.expand_dims(X_test_scaled, axis=2)

# Select a few samples for testing
num_samples = 5
sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
sample_X = X_test_reshaped[sample_indices]
sample_y = y_test.iloc[sample_indices]

# Make predictions on the samples
pred_probs = model.predict(sample_X)
pred_labels = np.argmax(pred_probs, axis=1)

# Print the results
print(f"Testing {num_samples} specific test cases:")
for i in range(num_samples):
    print(f"Sample {i+1}:")
    print(f"  Features: {sample_X[i].flatten()}")
    print(f"  True Label: {sample_y.iloc[i]}")
    print(f"  Predicted Label: {pred_labels[i]}")
    print("---")
