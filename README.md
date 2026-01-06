# Boosting Models vs Neural Networks for Credit Risk Prediction

This project compares **neural-network models** against **boosting models** for predicting vehicle loan defaults. The repo is organized around a simple, collaboration-friendly workflow:

- A shared processed dataset in `data/processed/`
- A shared split file in `data/splits/`
- Notebook-driven training and evaluation

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models Compared](#models-compared)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project evaluates and compares the performance of traditional gradient boosting algorithms against modern deep learning approaches for credit risk assessment in the context of vehicle loan defaults. The goal is to identify which modeling approach provides the best predictive performance while considering factors such as:

- **Accuracy**: Overall prediction accuracy and error rates
- **Speed**: Training and inference time
- **Interpretability**: Model explainability and feature importance
- **Robustness**: Performance on imbalanced datasets
- **Scalability**: Ability to handle large datasets

## 📊 Dataset

The project uses a **Vehicle Loan Default Dataset** containing historical information about borrowers and their loan repayment behavior.

**Source**: https://www.kaggle.com/code/vineetverma/vehicle-loan-default-prediction/input

### Dataset Features

Typical features include:
- **Demographic Information**: Age, income, employment status
- **Loan Characteristics**: Loan amount, term, interest rate, down payment
- **Vehicle Information**: Vehicle age, price, type
- **Credit History**: Previous defaults, credit score, number of inquiries
- **Behavioral Data**: Payment history, account age

### Target Variable

- **Binary Classification**: Default (1) vs Non-Default (0)

### Important Note

`data/test.csv` is **unlabeled** (no `LOAN_DEFAULT`). For model evaluation we create a labeled holdout split from `data/train.csv` and store it in `data/splits/split_uniqueid.csv`.

### Data Statistics

- **Train size**: 233,154 rows
- **Columns**: 41 (40 features + `LOAN_DEFAULT`)
- **Class distribution**: imbalanced
   - Defaults (`LOAN_DEFAULT=1`): 50,611 (~21.71%)
   - Non-defaults (`LOAN_DEFAULT=0`): 182,543 (~78.29%)

## 🤖 Models Compared

### Boosting Models

1. **XGBoost (Extreme Gradient Boosting)**
   - Regularized boosting framework
   - Handles missing values automatically
   - Built-in feature importance

2. **LightGBM (Light Gradient Boosting Machine)**
   - Leaf-wise tree growth
   - Faster training speed
   - Lower memory usage

3. **CatBoost (Categorical Boosting)**
   - Native categorical feature support
   - Ordered boosting to avoid overfitting
   - Robust to hyperparameter tuning

### Neural Network Models

1. **Baseline MLP (Feedforward NN)**
   - Multi-layer perceptron architecture
   - ReLU activation functions
   - Dropout for regularization

2. **Improved MLP**
   - Deeper architecture vs baseline
   - Batch normalization
   - Regularization (dropout / weight decay)

## 📁 Project Structure

```
.
├── data/
│   ├── train.csv                # Labeled training data (includes LOAN_DEFAULT)
│   ├── test.csv                 # Unlabeled data (no LOAN_DEFAULT)
│   ├── processed/
│   │   └── loan_processed.csv   # Shared processed dataset (from notebook)
│   └── splits/
│       └── split_uniqueid.csv   # Shared split mapping by UNIQUEID
├── notebooks/
│   ├── 01_preprocessing.ipynb   # Creates loan_processed.csv + split_uniqueid.csv
│   ├── 02_neural_networks.ipynb # Trains baseline + improved MLP; exports predictions
│   └── 04_model_comparison.ipynb# Compares NN models; boosting optional later
├── models/                     # Saved model files
├── results/
│   ├── nn_baseline_test_preds.csv
│   └── nn_improved_test_preds.csv
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/shadow-27/Boosting-Models-vs-Neural-networks-for-Credit-risk-.git
   cd Boosting-Models-vs-Neural-networks-for-Credit-risk-
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

Boosting libraries (e.g., XGBoost/LightGBM/CatBoost) are **optional** and can be installed separately if/when needed.

## 💻 Usage

This repo is notebook-driven. Run the notebooks in this order:

1. **Preprocess + create split**: `notebooks/01_preprocessing.ipynb`
   - Outputs:
     - `data/processed/loan_processed.csv`
     - `data/splits/split_uniqueid.csv`

2. **Train neural nets + export predictions**: `notebooks/02_neural_networks.ipynb`
   - Trains a baseline MLP and an improved MLP
   - Exports prediction files to `results/` (expected columns: `UNIQUEID`, `y_true`, `y_prob`)

3. **Compare models (NN-only by default)**: `notebooks/04_model_comparison.ipynb`
   - Compares baseline vs improved MLP
   - Boosting is intentionally **optional**: set `USE_BOOST = True` once there is  boosting models and results added in `results/boosting_best_test_preds.csv`

## 📈 Results

### Performance Comparison

Metrics below are computed on the labeled holdout set (from `data/splits/split_uniqueid.csv`). Precision/Recall/F1 are reported at a common threshold **t = 0.50**.

| Model | AUC-ROC | Precision@0.50 | Recall@0.50 | F1@0.50 |
|-------|--------:|---------------:|------------:|--------:|
| Baseline MLP | 0.650593 | 0.290820 | 0.668862 | 0.405381 |
| Improved MLP | 0.648562 | 0.294045 | 0.626976 | 0.400336 |
| XGBoost | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD |
| CatBoost | TBD | TBD | TBD | TBD |

*Note: Run `notebooks/04_model_comparison.ipynb` to generate the latest metrics + ROC plot.*

### Key Findings

- **Best NN (AUC-ROC / F1@0.50)**: Baseline MLP (slightly higher than Improved MLP on this split)
- **Boosting comparison**: TBD (to be added once boosting predictions are available)

## 📊 Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of predictions
- **Feature Importance**: Most influential features (for tree-based models)


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: https://www.kaggle.com/code/vineetverma/vehicle-loan-default-prediction/input

---

**Note**: This is an active research project. Results and documentation will be updated as the analysis progresses.