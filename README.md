# 📈 Stock Market Price Forecasting

A machine learning project that analyzes historical stock market data and builds predictive models to forecast future closing prices.

---

## 📌 Project Overview

This project focuses on predicting future stock closing prices using historical market data. It combines exploratory data analysis (EDA), time-series decomposition, feature engineering, and regression-based machine learning models to provide stakeholders and investors with reliable price estimates.

---

## 🎯 Objectives

- **Analytical Objective:** Analyze historical stock market data to identify trends and patterns.
- **Business Objective:** Help stakeholders and investors estimate future stock closing prices based on historical market behavior.

---

## 📁 Project Structure

```
Stock Market Forecasting/
│
├── Stock_Price_Forecasting.ipynb   # Main Jupyter Notebook with full analysis pipeline
├── HistoricalQuotes.xls            # Raw historical stock price data
├── ridge_stock_model.pkl           # Trained Ridge Regression model (saved)
├── stock_feature_scaler.pkl        # StandardScaler fitted on training features (saved)
└── README.md                       # Project documentation
```

---

## 📊 Dataset

The dataset (`HistoricalQuotes.xls`) contains **253 daily trading records** spanning from **August 2018 to August 2019**, with the following features:

| Column   | Description                         |
|----------|-------------------------------------|
| `date`   | Trading date                        |
| `close`  | Closing price of the stock          |
| `volume` | Number of shares traded             |
| `open`   | Opening price of the stock          |
| `high`   | Highest price during the trading day|
| `low`    | Lowest price during the trading day |

### Data Summary

- **Records:** 253 total → 252 after removing non-trading days (weekends)
- **Price Range:** $142.19 – $232.07 (close)
- **Average Close:** ~$193.09
- **No missing values or duplicates** found

---

## 🔍 Workflow & Methodology

### 1. Data Loading & Inspection
- Loaded data from CSV
- Inspected shape, data types, missing values, and duplicates

### 2. Data Preprocessing
- Parsed and converted `date` column to `datetime` format
- Sorted data chronologically
- Set `date` as the index
- Removed weekend/non-trading day entries (1 record filtered out)

### 3. Exploratory Data Analysis (EDA)
- Plotted closing price trends over time (daily and weekly)
- Performed **Seasonal Decomposition** using `statsmodels`:
  - Identified trend, seasonality, and residual components
- Analyzed volume distribution and OHLC patterns

### 4. Feature Engineering
- Created **lagged features** (previous closing prices)
- Computed **rolling statistics** (moving averages, rolling std)
- Extracted time-based features (e.g., day of week, month)
- Applied **StandardScaler** for feature normalization

### 5. Model Training & Evaluation
Two models were explored and evaluated:

| Model                    | Description                                      |
|--------------------------|--------------------------------------------------|
| **Random Forest Regressor** | Ensemble tree-based model; used as a baseline |
| **Ridge Regression**        | Regularized linear model; selected final model |

Evaluation metrics used:
- **MAE** – Mean Absolute Error
- **MSE** – Mean Squared Error
- **RMSE** – Root Mean Squared Error
- **R² Score** – Coefficient of Determination

### 6. Model Persistence
- The trained **Ridge Regression model** is saved as `ridge_stock_model.pkl`
- The **feature scaler** is saved as `stock_feature_scaler.pkl` for consistent preprocessing during inference

---

## 🛠️ Tech Stack

| Tool/Library     | Purpose                                      |
|------------------|----------------------------------------------|
| Python           | Programming language                         |
| Pandas           | Data manipulation and analysis               |
| NumPy            | Numerical computations                       |
| Matplotlib       | Data visualization                           |
| Statsmodels      | Time-series decomposition                    |
| Scikit-learn     | Machine learning models and preprocessing    |
| Joblib           | Model serialization (save/load `.pkl` files) |
| Jupyter Notebook | Interactive development and visualization    |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed. Install the required libraries:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn joblib jupyter
```

### Running the Notebook

1. Clone or download this repository.
2. Place `HistoricalQuotes.xls` in the same directory as the notebook.
3. Launch Jupyter Notebook:

```bash
jupyter notebook Stock_Price_Forecasting.ipynb
```

4. Run all cells sequentially to reproduce the full analysis and model training.

### Loading the Saved Model

To use the pre-trained model for inference:

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("ridge_stock_model.pkl")
scaler = joblib.load("stock_feature_scaler.pkl")

# Prepare your feature vector (same features used in training)
X_new = scaler.transform(your_feature_dataframe)
prediction = model.predict(X_new)
print(f"Predicted closing price: {prediction[0]:.2f}")
```

---

## 📈 Key Insights

- The stock price exhibits a **clear trend** over the one-year period, with noticeable seasonality patterns.
- **Ridge Regression** outperformed the baseline and was selected as the final model due to its stability and generalization ability.
- Feature engineering (lagged prices and rolling statistics) significantly contributed to prediction accuracy.

---

## ⚠️ Limitations

- The model is trained on **one year of data** (Aug 2018 – Aug 2019) and may not generalize well to other market conditions.
- External factors (news, earnings reports, macroeconomic events) are **not captured** in the OHLCV data.
- This model is intended for **educational and research purposes only** and should **not** be used for real financial decisions.

---

## 📄 License

This project is for academic and educational use only.
