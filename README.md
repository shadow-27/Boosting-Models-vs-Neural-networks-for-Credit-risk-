# Boosting Models vs Neural Networks for Credit Risk Prediction

A comprehensive comparative analysis of gradient boosting models (XGBoost, LightGBM, CatBoost) and deep neural networks for predicting vehicle loan defaults.

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

### Dataset Features

Typical features include:
- **Demographic Information**: Age, income, employment status
- **Loan Characteristics**: Loan amount, term, interest rate, down payment
- **Vehicle Information**: Vehicle age, price, type
- **Credit History**: Previous defaults, credit score, number of inquiries
- **Behavioral Data**: Payment history, account age

### Target Variable

- **Binary Classification**: Default (1) vs Non-Default (0)

### Data Statistics

- **Size**: TBD (To be updated with actual dataset)
- **Features**: TBD
- **Class Distribution**: Typically imbalanced (minority defaults)

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

1. **Feedforward Neural Network (FNN)**
   - Multi-layer perceptron architecture
   - ReLU activation functions
   - Dropout for regularization

2. **Deep Neural Network (DNN)**
   - Deeper architecture with multiple hidden layers
   - Batch normalization
   - Advanced optimization techniques

## 📁 Project Structure

```
.
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and preprocessed data
│   └── splits/                 # Train/validation/test splits
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_boosting_models.ipynb
│   ├── 04_neural_networks.ipynb
│   └── 05_model_comparison.ipynb
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── boosting_models.py
│   │   └── neural_networks.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│       └── helpers.py
├── models/                     # Saved model files
├── results/
│   ├── figures/               # Visualizations
│   └── reports/               # Performance reports
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

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Libraries

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
tensorflow>=2.8.0  # or pytorch>=1.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 💻 Usage

### Data Preprocessing

```python
from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_data

# Load data
df = load_dataset('data/raw/loan_data.csv')

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(df)
```

### Training Boosting Models

```python
from src.models.boosting_models import train_xgboost, train_lightgbm, train_catboost

# Train XGBoost
xgb_model = train_xgboost(X_train, y_train)

# Train LightGBM
lgb_model = train_lightgbm(X_train, y_train)

# Train CatBoost
cat_model = train_catboost(X_train, y_train)
```

### Training Neural Networks

```python
from src.models.neural_networks import build_fnn, build_dnn

# Build and train FNN
fnn_model = build_fnn(input_dim=X_train.shape[1])
fnn_model.fit(X_train, y_train, epochs=50, batch_size=32)

# Build and train DNN
dnn_model = build_dnn(input_dim=X_train.shape[1])
dnn_model.fit(X_train, y_train, epochs=100, batch_size=64)
```

### Model Evaluation

```python
from src.evaluation.metrics import evaluate_model

# Evaluate all models
for model_name, model in models.items():
    metrics = evaluate_model(model, X_test, y_test)
    print(f"{model_name}: {metrics}")
```

## 📈 Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| XGBoost | TBD | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD | TBD |
| CatBoost | TBD | TBD | TBD | TBD | TBD | TBD |
| FNN | TBD | TBD | TBD | TBD | TBD | TBD |
| DNN | TBD | TBD | TBD | TBD | TBD | TBD |

*Note: Results will be updated after model training and evaluation*

### Key Findings

- **Best Overall Performance**: TBD
- **Fastest Training**: TBD
- **Most Interpretable**: TBD
- **Best for Imbalanced Data**: TBD

## 📊 Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of predictions
- **Feature Importance**: Most influential features (for tree-based models)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [To be added]
- Inspiration from various credit risk modeling research papers
- Open-source machine learning community

## 📞 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This is an active research project. Results and documentation will be updated as the analysis progresses.