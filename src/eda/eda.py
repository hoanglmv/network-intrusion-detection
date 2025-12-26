
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda():
    """
    Performs Exploratory Data Analysis (EDA) on the first 10 CSV files
    from the data/CIC2023 directory.
    """
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "CIC2023")
    figure_dir = os.path.join(project_root, "figures")

    # Create figure directory if it doesn't exist
    os.makedirs(figure_dir, exist_ok=True)

    # Get the first 10 CSV files
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))[:10]

    if not csv_files:
        print("No CSV files found in data/CIC2023 directory.")
        return

    # Concatenate files into a single DataFrame
    print("Loading and concatenating the first 10 CSV files...")
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print("Data loaded successfully.")

    # --- Basic Information ---
    print("\n--- Basic Data Information ---")
    print(f"Shape of the combined DataFrame: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types and non-null counts:")
    df.info()
    print("\nSummary statistics for numerical columns:")
    print(df.describe())

    # Clean up column names by stripping leading/trailing spaces
    df.columns = df.columns.str.strip()

    # --- Label Distribution ---
    print("\n--- Analyzing Label Distribution ---")
    label_col = 'label' # Adjusted column name after stripping spaces
    if label_col in df.columns:
        plt.figure(figsize=(12, 8))
        sns.countplot(y=df[label_col], order=df[label_col].value_counts().index)
        plt.title('Distribution of Network Attack Labels')
        plt.xlabel('Count')
        plt.ylabel('Label')
        plt.tight_layout()
        label_dist_path = os.path.join(figure_dir, "label_distribution.png")
        plt.savefig(label_dist_path)
        plt.close()
        print(f"Saved label distribution plot to: {label_dist_path}")
    else:
        print(f"'{label_col}' column not found!")

    # --- Correlation Analysis ---
    print("\n--- Analyzing Feature Correlation ---")
    # Select only numeric columns for correlation matrix
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Handle potential infinity values if any
    df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_numeric = df_numeric.dropna(axis=1, how='any') # Drop columns with NaN values for simplicity

    if not df_numeric.empty:
        correlation_matrix = df_numeric.corr()
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, cmap='viridis', annot=False)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        corr_heatmap_path = os.path.join(figure_dir, "correlation_heatmap.png")
        plt.savefig(corr_heatmap_path)
        plt.close()
        print(f"Saved correlation heatmap to: {corr_heatmap_path}")
    else:
        print("No numeric columns available for correlation analysis after cleaning.")

    print("\nEDA script execution finished.")

if __name__ == '__main__':
    perform_eda()
