
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
    df.columns = df.columns.str.strip().str.replace(' ', '_')

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

    # --- Numerical Feature Distribution ---
    print("\n--- Analyzing Numerical Feature Distributions ---")
    # Select some key numerical features to plot
    key_numerical_features = [
        'flow_duration', 'Header_Length', 'Protocol_Type', 'Duration', 'Rate', 'Srate'
    ]
    # Filter out features that are not in the dataframe
    key_numerical_features = [feat for feat in key_numerical_features if feat in df.columns]

    if key_numerical_features:
        plot_numerical_distributions(df, key_numerical_features, figure_dir)
        plot_numerical_boxplots(df, key_numerical_features, 'label', figure_dir)
    else:
        print("None of the selected key numerical features are in the DataFrame.")

    print("\nEDA script execution finished.")


def plot_numerical_distributions(df, features, figure_dir):
    """
    Plots and saves the distribution of numerical features.
    """
    print(f"Plotting distributions for: {', '.join(features)}")
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=50)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        # Use log scale for features with high skewness
        if df[feature].skew() > 3:
            plt.yscale('log')
            plt.title(f'Distribution of {feature} (Log Scale)')
        plt.tight_layout()
        dist_path = os.path.join(figure_dir, f"{feature}_distribution.png")
        plt.savefig(dist_path)
        plt.close()
        print(f"Saved {feature} distribution plot to: {dist_path}")


def plot_numerical_boxplots(df, features, target_col, figure_dir):
    """
    Plots and saves boxplots of numerical features against a target column.
    """
    print(f"Plotting boxplots for: {', '.join(features)}")
    for feature in features:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=target_col, y=feature, data=df)
        plt.title(f'{feature} by {target_col}')
        plt.xlabel(target_col)
        plt.ylabel(feature)
        plt.xticks(rotation=45)
        # Use log scale for features with high skewness
        if df[feature].skew() > 3:
            plt.yscale('log')
            plt.title(f'{feature} by {target_col} (Log Scale)')
        plt.tight_layout()
        boxplot_path = os.path.join(figure_dir, f"{feature}_vs_{target_col}_boxplot.png")
        plt.savefig(boxplot_path)
        plt.close()
        print(f"Saved {feature} vs {target_col} boxplot to: {boxplot_path}")


if __name__ == '__main__':
    perform_eda()
