# Network Intrusion Detection Project

This project implements and evaluates several machine learning models for network intrusion detection. The goal is to classify network traffic into different attack types or normal behavior based on various network flow features.

## Project Structure

```
intrusion-network/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── CIC2023/            # Raw dataset (if applicable)
│   └── processed/          # Processed data files (e.g., data1.csv, train_data.csv, etc.)
├── pretrained/             # Trained models and label encoders
│   ├── label_encoder.joblib
│   ├── linear_regression_model.joblib
│   ├── random_forest_model.joblib
│   ├── xgboost_model.joblib
│   └── rnn_model.h5
│   └── cnn_model.h5
└── src/
    ├── data/               # Data preprocessing and analysis scripts
    │   ├── split_data.py
    │   └── data_analysis.ipynb
    └── model/              # Model training and testing scripts/notebooks
        ├── cnn/
        │   ├── cnn.ipynb
        │   ├── train_cnn.py
        │   └── test.ipynb
        ├── linear_regression/
        │   ├── linear_regression.ipynb
        │   ├── train_linear_regression.py
        │   └── test.ipynb
        ├── random_forest/
        │   ├── random_forest.ipynb
        │   ├── train_random_forest.py
        │   └── test.ipynb
        ├── rnn/
        │   ├── rnn.ipynb
        │   ├── train_rnn.py
        │   └── test.ipynb
        └── xgboost/
            ├── xgboost.ipynb
            ├── train_xgboost.py
            └── test.ipynb
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd intrusion-network
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
    Ensure `data1.csv` is placed in the `data/processed/` directory. This file is expected to be the preprocessed dataset used for training.

## Usage

### 1. Data Splitting

First, split the `data1.csv` into training, validation, and testing sets. This script will also save the `LabelEncoder`.

```bash
python src/data/split_data.py
```

This will generate `train_data.csv`, `val_data.csv`, and `test_data.csv` in the `data/processed/` directory, and `label_encoder.joblib` in the `pretrained/` directory.

### 2. Data Analysis and Visualization

To explore the dataset and visualize its features, open and run the `data_analysis.ipynb` notebook:

```bash
jupyter notebook src/data/data_analysis.ipynb
```

### 3. Model Training

Train each model by running its respective training script:

*   **Linear Regression:**
    ```bash
    python src/model/linear_regression/train_linear_regression.py
    ```

*   **Random Forest:**
    ```bash
    python src/model/random_forest/train_random_forest.py
    ```

*   **CNN:**
    ```bash
    python src/model/cnn/train_cnn.py
    ```

*   **XGBoost:**
    ```bash
    python src/model/xgboost/train_xgboost.py
    ```

*   **RNN:**
    ```bash
    python src/model/rnn/train_rnn.py
    ```

Each script will train the model, evaluate it on the test set, print a classification report, and save the trained model (e.g., `linear_regression_model.joblib`, `random_forest_model.joblib`, `xgboost_model.joblib`, `cnn_model.h5`, `rnn_model.h5`) to the `pretrained/` directory.

### 4. Model Testing and Evaluation

To test and evaluate each trained model, open and run its respective `test.ipynb` notebook:

*   **Linear Regression:**
    ```bash
    jupyter notebook src/model/linear_regression/test.ipynb
    ```

*   **Random Forest:**
    ```bash
    jupyter notebook src/model/random_forest/test.ipynb
    ```

*   **CNN:**
    ```bash
    jupyter notebook src/model/cnn/test.ipynb
    ```

*   **XGBoost:**
    ```bash
    jupyter notebook src/model/xgboost/test.ipynb
    ```

*   **RNN:**
    ```bash
    jupyter notebook src/model/rnn/test.ipynb
    ```

These notebooks will load the saved models, make predictions on the test data, and display performance metrics including accuracy, classification reports, and confusion matrices.

## Results Summary (Example)

| Model             | Accuracy (Test Set) |
| :---------------- | :------------------ |
| Linear Regression | ~--.--%             |
| Random Forest     | ~99.87%             |
| CNN               | ~81.04%             |
| XGBoost           | ~99.93%             |
| RNN               | ~85.70%             |

*(Note: Actual results may vary slightly due to random seeds or environment differences.)*
