# Stock Market Time-Series Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14%2B-green.svg)](https://www.statsmodels.org/)

An end-to-end, technically correct, and presentation-ready stock market time-series forecasting system built with Python. This project analyzes historical stock market data (`HistoricalQuotes.csv`), constructs statistical lag and rolling window features with strict data-leakage prevention, benchmarks regression models against a Naive Baseline, and generates a 10-trading-day recursive future forecast.

---

## 📌 Problem Statement

Stock market prices exhibit high volatility and strong chronological autocorrelation. The objective of this project is to analyze historical market patterns and predict the target variable **`Close`** (Closing Price) using a structured time-series machine learning pipeline.

---

## 🎯 Business Objective

Provide a robust data-science workflow that translates historical stock quotes into actionable time-series price estimations. The project is designed to be easily understandable for academic and internship project presentations while implementing rigorous data-science engineering practices.

---

## 📊 Dataset Characteristics

- **Primary Source:** `data/HistoricalQuotes.csv`
- **Total Records:** 253 valid trading rows spanning `2018-08-09` to `2019-08-10` (~1 full year)
- **Columns:** `date`, `close`, `volume`, `open`, `high`, `low`
- **Target Variable:** `close` (Price range: **\$142.19** to **\$232.07**, Mean: **\$193.09**)
- **Data Quality:** 0 missing values, 0 duplicate rows
- **Missing Day Analysis:** Out of 114 missing calendar days (from 367 total calendar days), 104 are standard weekends and **10 correspond to official US market holidays**:
  - `2018-09-03` (Labor Day)
  - `2018-11-22` (Thanksgiving Day)
  - `2018-12-05` (National Day of Mourning - Pres. George H.W. Bush)
  - `2018-12-25` (Christmas Day)
  - `2019-01-01` (New Year's Day)
  - `2019-01-21` (Martin Luther King Jr. Day)
  - `2019-02-18` (Presidents' Day)
  - `2019-04-19` (Good Friday)
  - `2019-05-27` (Memorial Day)
  - `2019-07-04` (Independence Day)

---

## ⚙️ Complete Methodology Workflow

```text
Dataset (HistoricalQuotes.csv)
   ↓
Data Understanding & Quality Inspection
   ↓
Date-Time Parsing & Chronological Sorting
   ↓
Missing Time-Series Record Analysis & Holiday Mapping
   ↓
Frequency Analysis & Weekly/Monthly Resampling
   ↓
Exploratory Data Analysis (Price, Volume, Distribution)
   ↓
Trend Analysis (5-day, 10-day, 20-day Moving Averages)
   ↓
Time-Series Decomposition (Observed, Trend, Seasonality, Residuals)
   ↓
Leakage-Free Feature Engineering (Lags 1-20, Rolling Means/Stds, Returns)
   ↓
Chronological Train/Test Split (80% Train / 20% Test) & Training Scaler
   ↓
Naive Baseline Modeling (Tomorrow = Today's Close)
   ↓
Machine Learning Regression (Linear Regression, Random Forest, Gradient Boosting)
   ↓
Model Evaluation & Metric Comparison (MAE, RMSE, R², MAPE)
   ↓
Residual Diagnostic Analysis
   ↓
10-Trading-Day Recursive Future Forecasting
   ↓
Model & Visualization Serialization
```

---

## 🛠️ Technology Stack

- **Core & Logic:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`
- **Time-Series Analysis:** `statsmodels`
- **Model Serialization:** `joblib`
- **Interactive Notebooks:** `jupyter`

---

## 📈 Model Performance Results

Models were evaluated on unseen chronological test data (47 trading days, `2019-06-06` to `2019-08-10`):

| Model | MAE ($) | RMSE ($) | R² Score | MAPE (%) | Key Insights |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Naive Baseline** | 2.2683 | 2.9662 | 0.7291 | 1.1330% | Strong baseline reflecting random walk stock behavior |
| **Linear Regression (Primary)** | 2.3999 | 3.1595 | 0.6927 | 1.2036% | Best interpretable ML model; tracks price momentum cleanly |
| **Random Forest Regressor** | 2.8290 | 3.7076 | 0.5768 | 1.4180% | Slight overfitting on historical training bounds |
| **Gradient Boosting Regressor** | 3.2620 | 4.1183 | 0.4778 | 1.6341% | Higher variance on unseen out-of-time evaluation |

---

## 🔮 10-Trading-Day Future Forecast Table

Recursive multi-step forecast for the next 10 business trading days (excluding weekends):

| Date | Day | Predicted Close ($) |
| :--- | :--- | ---: |
| `2019-08-12` | Monday | **\$199.24** |
| `2019-08-13` | Tuesday | **\$199.38** |
| `2019-08-14` | Wednesday | **\$197.99** |
| `2019-08-15` | Thursday | **\$198.60** |
| `2019-08-16` | Friday | **\$197.36** |
| `2019-08-19` | Monday | **\$197.50** |
| `2019-08-20` | Tuesday | **\$196.51** |
| `2019-08-21` | Wednesday | **\$196.77** |
| `2019-08-22` | Thursday | **\$195.89** |
| `2019-08-23` | Friday | **\$196.23** |

---

## 🖼️ Generated Visualizations

All visual charts are saved automatically under `outputs/`:

1. **Price & Volume Distribution (`outputs/stock_price.png`):** Historical close prices, trading volume, and price histogram.
2. **Moving Averages (`outputs/moving_average.png`):** 5-day, 10-day, and 20-day moving average trend lines.
3. **Time-Series Decomposition (`outputs/decomposition.png`):** Additive breakdown of Observed, Trend, Seasonality (20-day period), and Residuals.
4. **Actual vs Predicted (`outputs/actual_vs_predicted.png`):** Test period actual stock prices plotted against Naive Baseline and ML model predictions.
5. **Residual Diagnostics (`outputs/residuals.png`):** Error scatter over time and residual error distribution centered near zero.
6. **Future Forecast (`outputs/future_forecast.png`):** Historical 30-day price trend seamlessly connected to the 10-day future prediction curve.

---

## 📁 Repository Structure

```text
Stock-Market-Forecasting/
│
├── data/
│   └── HistoricalQuotes.csv             # Primary dataset
│
├── notebooks/
│   └── stock_forecasting.ipynb          # Interactive 22-section Jupyter Notebook
│
├── src/
│   ├── __init__.py                       # Package initialization
│   ├── data_preprocessing.py            # Loading, datetime parsing, holiday analysis
│   ├── time_series_analysis.py          # Resampling, MA trends, seasonal decomposition
│   ├── feature_engineering.py           # Lag and rolling feature generation
│   ├── model_training.py                # Chronological split, scaling, training, metrics
│   └── forecasting.py                   # Multi-step recursive future forecasting
│
├── models/
│   └── stock_forecasting_model.pkl      # Saved trained model artifact & scaler
│
├── outputs/
│   ├── stock_price.png                  # EDA chart
│   ├── moving_average.png               # Trend chart
│   ├── decomposition.png                # Seasonal decomposition plot
│   ├── actual_vs_predicted.png          # Model comparison plot
│   ├── residuals.png                    # Residual diagnostics plot
│   └── future_forecast.png              # 10-day future forecast plot
│
├── requirements.txt                     # Project dependencies
├── README.md                            # Comprehensive documentation
└── run.py                               # Master end-to-end execution pipeline
```

---

## 🚀 Installation & Usage Guide

### 1. Prerequisites & Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/yayatipatil/Stock-Market-Forecasting.git
cd Stock-Market-Forecasting
pip install -r requirements.txt
```

### 2. Execute Master Pipeline Script

Run the complete pipeline end-to-end to preprocess data, train models, save artifacts, and generate all output plots:

```bash
python run.py
```

### 3. Launch Interactive Jupyter Notebook

Open and run the beginner-friendly step-by-step notebook:

```bash
jupyter notebook notebooks/stock_forecasting.ipynb
```

---

## 💡 How to Explain This Project in an Internship Presentation

When presenting this project to a mentor or examiner:
1. **Explain the Objective:** "We aim to predict tomorrow's stock closing price using historical OHLCV quotes."
2. **Highlight Data Quality:** "We verified date continuity across 367 calendar days and proved that all 10 missing business weekdays correspond to official US stock market holidays."
3. **Emphasize Leakage Prevention:** "All features—lags (1-20 days) and rolling averages (5-20 days)—were shifted by 1 day (`.shift(1)`), and scaling was fitted exclusively on the 80% training set to guarantee zero lookahead bias."
4. **Discuss Model Insights:** "Linear Regression achieved an MAE of ~\$2.40 (1.2% MAPE), performing close to the strong Naive Baseline. Complex non-linear tree models overfit historical training bounds, confirming that simpler linear models with lag features are superior for short-term financial time series."

---

## ⚠️ Limitations & Future Scope

### Current Limitations:
- Uses 1 year of daily historical quotes for a single ticker.
- Does not incorporate macro-economic indicators, company earnings reports, or financial news sentiment.

### Future Scope:
- Incorporate technical indicators (RSI, MACD, Bollinger Bands).
- Implement deep learning architectures (LSTM, GRU, Temporal Convolutional Networks).
- Expand to multi-stock portfolios and sentiment analysis using financial news feeds.
- Implement Walk-Forward (Expanding Window) cross-validation.

---

## ⚠️ Important Financial Disclaimer

> **DISCLAIMER:** This project is created strictly for **educational and presentation purposes**. Stock market predictions generated by this system are statistical estimates based on historical patterns and do **NOT** constitute financial advice, investment strategies, or guarantees of future market performance.
