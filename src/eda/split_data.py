
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define paths relative to the project root
output_dir = os.path.join(project_root, 'data', 'processed')
data_path = os.path.join(output_dir, 'data.csv')
df = pd.read_csv(data_path)

# Encode the 'label' column
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Split the data
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the split datasets
train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

print("Data split and saved successfully!")
