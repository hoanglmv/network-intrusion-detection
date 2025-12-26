# Network Intrusion Detection Project

This project implements and evaluates several machine learning models for network intrusion detection. The goal is to classify network traffic into different attack types or normal behavior based on various network flow features.

## Project Structure

```
network-intrusion-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── CIC2023/            # Raw dataset (if applicable)
│   └── processed/          # Processed data files (e.g., data.csv, train_data.csv, etc.)
├── trained/             # Trained models and training history
│   ├── cnn_history.csv
│   ├── cnn_model.h5
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   ├── rnn_history.csv
│   ├── rnn_model.h5
│   ├── xgboost_history.csv
│   └── xgboost_model.joblib
└── src/
    ├── pretrained/         # Scripts for data preprocessing
    │   └── split_data.py
    └── model/              # Model training and testing scripts/notebooks
        ├── cnn/
        │   ├── train_cnn.py
        │   └── test.ipynb
        ├── logistic_regression/
        │   ├── train_logistic_regression.py
        │   └── test.ipynb
        ├── random_forest/
        │   ├── train_random_forest.py
        │   └── test.ipynb
        ├── rnn/
        │   ├── train_rnn.py
        │   └── test.ipynb
        └── xgboost/
            ├── train_xgboost.py
            └── test.ipynb
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd network-intrusion-detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    Ensure `data.csv` is placed in the `data/processed/` directory. This file is expected to be the preprocessed dataset used for training.

## Usage

### 1. Data Splitting

First, split the `data.csv` into training, validation, and testing sets.

```bash
python src/pretrained/split_data.py
```

This will generate `train_data.csv`, `val_data.csv`, and `test_data.csv` in the `data/processed/` directory.

### 2. Model Training

Train each model by running its respective training script:

*   **CNN:**
    ```bash
    python src/model/cnn/train_cnn.py
    ```

*   **Logistic Regression:**
    ```bash
    python src/model/logistic_regression/train_logistic_regression.py
    ```

*   **Random Forest:**
    ```bash
    python src/model/random_forest/train_random_forest.py
    ```

*   **RNN:**
    ```bash
    python src/model/rnn/train_rnn.py
    ```

*   **XGBoost:**
    ```bash
    python src/model/xgboost/train_xgboost.py
    ```

Each script will train the model, evaluate it on the test set, print a classification report, and save the trained model to the `trained/` directory. For CNN, RNN and XGBoost, the training history (loss and accuracy per epoch) will also be saved.

### 3. Model Testing and Evaluation

To test and evaluate each trained model, open and run its respective `test.ipynb` notebook:

*   **CNN:**
    ```bash
    jupyter notebook src/model/cnn/test.ipynb
    ```

*   **Logistic Regression:**
    ```bash
    jupyter notebook src/model/logistic_regression/test.ipynb
    ```

*   **Random Forest:**
    ```bash
    jupyter notebook src/model/random_forest/test.ipynb
    ```

*   **RNN:**
    ```bash
    jupyter notebook src/model/rnn/test.ipynb
    ```

*   **XGBoost:**
    ```bash
    jupyter notebook src/model/xgboost/test.ipynb
    ```

These notebooks will load the saved models, make predictions on the test data, and display performance metrics including accuracy, classification reports, and confusion matrices.

## Results Summary

After retraining all models, the following table summarizes the performance on the test set.

| Model               | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :------------------ | :------- | :------------------- | :---------------- | :------------------ |
| Random Forest       | 1.00     | 1.00                 | 1.00              | 1.00                |
| XGBoost             | 1.00     | 1.00                 | 1.00              | 1.00                |
| CNN                 | 0.93     | 0.96                 | 0.93              | 0.93                |
| RNN                 | 0.86     | 0.89                 | 0.86              | 0.85                |
| Logistic Regression | 0.84     | 0.86                 | 0.84              | 0.83                |

**Note:** The tree-based models (Random Forest and XGBoost) achieved near-perfect scores, which may indicate overfitting but also demonstrates their high effectiveness on this dataset. The neural network models (CNN and RNN) also performed well, followed by Logistic Regression.
