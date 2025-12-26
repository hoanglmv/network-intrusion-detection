# src/eda/eda_data.py
import pandas as pd
import os

def perform_eda(file_path):
    """
    Performs basic Exploratory Data Analysis (EDA) on a given CSV file.

    Args:
        file_path (str): The path to the CSV file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Performing EDA on {file_path} ---")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print("\n--- Dataset Head ---")
    print(df.head())

    print("\n--- Dataset Info ---")
    df.info()

    print("\n--- Dataset Description ---")
    print(df.describe())

    print("\n--- Missing Values ---")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if missing_values.empty:
        print("No missing values found.")
    else:
        print(missing_values)
        print("\n--- Percentage of Missing Values ---")
        print((missing_values / len(df)) * 100)

    print("\n--- Unique Values in Each Column (Top 5 if many) ---")
    for column in df.columns:
        unique_count = df[column].nunique()
        print(f"Column '{column}': {unique_count} unique values")
        if unique_count > 20: # Display top 5 for columns with many unique values
            print(df[column].value_counts().head())
        else:
            print(df[column].value_counts())
        print("-" * 30)

if __name__ == "__main__":
    # Assuming the script is run from the project root or src/eda
    # Adjust the path as necessary if the execution context changes
    
    # Try to find the root directory dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'data.csv')
    
    perform_eda(processed_data_path)
