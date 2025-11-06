
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import os

# Define file paths
processed_data_dir = "D:\\vhproj\\network-intrusion-detection\\data\\processed"
train_data_path = os.path.join(processed_data_dir, "train_data.csv")
val_data_path = os.path.join(processed_data_dir, "val_data.csv")
test_data_path = os.path.join(processed_data_dir, "test_data.csv")
model_output_path = "D:\\vhproj\\network-intrusion-detection\\trained\\rnn_model.h5"

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

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape data for RNN (samples, timesteps, features)
# We will treat each feature as a timestep
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_val_reshaped = np.reshape(X_val_scaled, (X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Get number of classes
num_classes = len(np.unique(y_train))

# Build the RNN model
model = Sequential([
    SimpleRNN(64, input_shape=(X_train_reshaped.shape[1], 1), return_sequences=True),
    Dropout(0.2),
    SimpleRNN(32),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train_reshaped, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_data=(X_val_reshaped, y_val),
                    verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=1)
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Loss: {loss:.4f}')

# Predictions and classification report
y_pred_probs = model.predict(X_test_reshaped)
y_pred = np.argmax(y_pred_probs, axis=1)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('D:\\vhproj\\network-intrusion-detection\\trained\\rnn_history.csv', index=False)

# Save the model
model.save(model_output_path)
print(f"Model saved to {model_output_path}")
