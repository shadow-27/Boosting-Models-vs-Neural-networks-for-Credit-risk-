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
│   ├── 03_boosting.ipynb         # Trains boosting models; exports predictions
│   └── 04_model_comparison_with_boosting.ipynb # Final comparison (boosting + NN)
├── models/                     # Saved model files
├── results/
│   ├── nn_baseline_test_preds.csv
│   ├── nn_improved_test_preds.csv
│   ├── boosting_xgb_test_preds.csv
│   ├── lgbm_test_preds.csv
│   └── boosting_cat_test_preds.csv
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

Boosting libraries are installed separately (only needed for `notebooks/03_boosting.ipynb`):

```bash
pip install xgboost lightgbm catboost
```

## 💻 Usage

This repo is notebook-driven. Run the notebooks in this order:

1. **Preprocess + create split**: `notebooks/01_preprocessing.ipynb`
   - Outputs:
     - `data/processed/loan_processed.csv`
     - `data/splits/split_uniqueid.csv`

2. **Train neural nets + export predictions**: `notebooks/02_neural_networks.ipynb`
   - Trains a baseline MLP and an improved MLP
   - Exports prediction files to `results/` (expected columns: `UNIQUEID`, `y_true`, `y_prob`)

3. **Train boosting models + export predictions**: `notebooks/03_boosting.ipynb`
   - Trains boosting models (XGBoost / LightGBM / CatBoost)
   - Exports prediction files to `results/` (expected columns: `UNIQUEID`, `y_true`, `y_prob`)

4. **Final comparison (boosting + NN)**: `notebooks/04_model_comparison_with_boosting.ipynb`
   - Loads the exported prediction CSVs for all models
   - Produces a metrics table and ROC curve plots

## 📈 Results

### Performance Comparison

Metrics below are computed on the labeled holdout set (from `data/splits/split_uniqueid.csv`). Threshold-based metrics are reported at a common threshold **t = 0.50**.

| Model | AUC-ROC | AUC-PR | Log Loss | Brier | Accuracy | Balanced Acc | Precision@0.50 | Recall@0.50 | Specificity@0.50 | F1@0.50 |
|-------|--------:|-------:|---------:|------:|---------:|-------------:|---------------:|------------:|-----------------:|--------:|
| Boosting - CatBoost | 0.666400 | 0.344419 | 0.493343 | 0.159949 | 0.783925 | 0.507541 | 0.568627 | 0.019099 | 0.995983 | 0.036957 |
| Boosting - XGBoost | 0.666278 | 0.343337 | 0.493397 | 0.159987 | 0.783525 | 0.508142 | 0.534426 | 0.021470 | 0.994814 | 0.041281 |
| Boosting - LightGBM | 0.665569 | 0.342976 | 0.493600 | 0.160064 | 0.783182 | 0.506828 | 0.516605 | 0.018440 | 0.995216 | 0.035610 |
| NN - Baseline MLP | 0.650593 | 0.325208 | 0.654173 | 0.232047 | 0.574055 | 0.608315 | 0.290820 | 0.668862 | 0.547769 | 0.405381 |
| NN - Improved MLP | 0.648562 | 0.323609 | 0.650185 | 0.229896 | 0.592269 | 0.604811 | 0.294045 | 0.626976 | 0.582646 | 0.400336 |

Confusion-matrix counts (**t = 0.50**):

| Model | TP | FP | TN | FN |
|-------|---:|---:|---:|---:|
| Boosting - CatBoost | 145 | 110 | 27272 | 7447 |
| Boosting - XGBoost | 163 | 142 | 27240 | 7429 |
| Boosting - LightGBM | 140 | 131 | 27251 | 7452 |
| NN - Baseline MLP | 5078 | 12383 | 14999 | 2514 |
| NN - Improved MLP | 4760 | 11428 | 15954 | 2832 |

*Note: Run `notebooks/04_model_comparison_with_boosting.ipynb` to generate the latest metrics + ROC/PR plots.*

### Key Findings

- **Best overall ranking (ROC-AUC / PR-AUC):** Boosting models lead, with **CatBoost** best ROC-AUC (**0.6664**) and best PR-AUC (**0.3444**) on this split.
- **Best probability quality (Log Loss / Brier):** Boosting models are substantially better (Log Loss ~**0.493**, Brier ~**0.160**) than the MLPs (Log Loss ~**0.65**, Brier ~**0.23**).
- **Best F1 at t = 0.50:** **NN - Baseline MLP** (**0.4054**).
- **Threshold matters:** at **t = 0.50**, boosting models have very high precision but **very low recall (~2%)** because their predicted probabilities are rarely above 0.5. In practice, the operating threshold should be tuned on the validation set to match the business objective.



## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: https://www.kaggle.com/code/vineetverma/vehicle-loan-default-prediction/input

---

**Note**: This is an active research project. Results and documentation will be updated as the analysis progresses.