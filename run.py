"""
Master Pipeline Script for Stock Market Time-Series Forecasting

Runs the end-to-end time-series forecasting workflow:
1. Preprocessing & Datetime Parsing
2. Missing Record Analysis & Holiday Mapping
3. Resampling & Visual Exploratory Data Analysis
4. Moving Averages & Trend Analysis
5. Seasonal Decomposition
6. Feature Engineering (Lags, Rolling Stats, Returns)
7. Chronological Train/Test Split & Leakage-Free Scaling
8. Naive Baseline & Machine Learning Model Training
9. Model Evaluation & Comparison Table
10. Residual Analysis
11. 10-Day Recursive Future Forecast
12. Model & Plot Serialization
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to system path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from data_preprocessing import load_raw_data, clean_and_sort_data, analyze_missing_records
from time_series_analysis import resample_data, plot_eda, calculate_moving_averages, decompose_series
from feature_engineering import build_all_features, prepare_feature_matrix
from model_training import (
    chronological_split,
    train_and_evaluate_models,
    plot_actual_vs_predicted,
    plot_residuals,
    save_best_model
)
from forecasting import recursive_future_forecast


def run_pipeline():
    print("=" * 75)
    print("      STOCK MARKET TIME-SERIES FORECASTING PIPELINE")
    print("=" * 75)

    root_dir = Path(__file__).resolve().parent
    data_path = root_dir / "data" / "HistoricalQuotes.csv"
    output_dir = root_dir / "outputs"
    models_dir = root_dir / "models"

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # 1. Data Preprocessing & Date-Time Processing
    # ----------------------------------------------------
    print("\n[Step 1] Loading and Preprocessing Raw Dataset...")
    raw_df = load_raw_data(data_path)
    clean_df = clean_and_sort_data(raw_df)

    print(f"  • Total valid records loaded: {len(clean_df)}")
    print(f"  • Date range: {clean_df['date'].min().strftime('%Y-%m-%d')} to {clean_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  • Close price range: ${clean_df['close'].min():.2f} to ${clean_df['close'].max():.2f} (Mean: ${clean_df['close'].mean():.2f})")

    # ----------------------------------------------------
    # 2. Missing Record Analysis
    # ----------------------------------------------------
    print("\n[Step 2] Analyzing Missing Time-Series Records & Market Closures...")
    missing_info = analyze_missing_records(clean_df)
    print(f"  • Calendar day span: {missing_info['total_calendar_days']} days")
    print(f"  • Expected business trading days: {missing_info['expected_business_days']}")
    print(f"  • Non-trading business days (Holidays): {missing_info['missing_business_days_count']}")
    print("  • Identified Market Holidays:")
    for h in missing_info["missing_business_days"]:
        print(f"      - {h['date']} ({h['day_name']}): {h['reason']}")
    if missing_info["weekend_records_count"] > 0:
        print(f"  • Documented Weekend Record: Saturday {missing_info['weekend_records'][0]['date'].strftime('%Y-%m-%d')} (Close: ${missing_info['weekend_records'][0]['close']:.2f})")

    # ----------------------------------------------------
    # 3. Resampling & Visual Time-Series EDA
    # ----------------------------------------------------
    print("\n[Step 3] Resampling and Generating Exploratory Data Analysis Plots...")
    resampled = resample_data(clean_df)
    print(f"  • Weekly aggregated records: {len(resampled['weekly'])}")
    print(f"  • Monthly aggregated records: {len(resampled['monthly'])}")

    p_eda = plot_eda(clean_df, output_dir)
    print(f"  • Saved EDA plot: [outputs/stock_price.png]({p_eda})")

    df_ma, p_ma = calculate_moving_averages(clean_df, output_dir)
    print(f"  • Saved Moving Averages plot: [outputs/moving_average.png]({p_ma})")

    decomp, p_dec = decompose_series(clean_df, period=20, output_dir=output_dir)
    print(f"  • Saved Decomposition plot: [outputs/decomposition.png]({p_dec})")

    # ----------------------------------------------------
    # 4. Feature Engineering
    # ----------------------------------------------------
    print("\n[Step 4] Engineering Historical Lag & Rolling Statistics Features...")
    feat_df = build_all_features(clean_df)
    prep_df = prepare_feature_matrix(feat_df)
    print(f"  • Total complete observation rows after feature creation: {len(prep_df)}")

    # ----------------------------------------------------
    # 5. Chronological Train/Test Split & Model Training
    # ----------------------------------------------------
    print("\n[Step 5] Splitting Data Chronologically (80% Train / 20% Test) & Training Models...")
    X_tr_s, X_te_s, y_tr, y_te, d_tr, d_te, scaler, feat_cols = chronological_split(prep_df, train_ratio=0.8)

    print(f"  • Training samples: {len(y_tr)} ({str(pd.to_datetime(d_tr[0]))[:10]} to {str(pd.to_datetime(d_tr[-1]))[:10]})")
    print(f"  • Testing samples:  {len(y_te)} ({str(pd.to_datetime(d_te[0]))[:10]} to {str(pd.to_datetime(d_te[-1]))[:10]})")

    results_df, models_dict, preds_dict = train_and_evaluate_models(X_tr_s, X_te_s, y_tr, y_te, prep_df, train_ratio=0.8)

    print("\n" + "-" * 60)
    print("               MODEL PERFORMANCE EVALUATION TABLE")
    print("-" * 60)
    print(results_df.to_string(index=False))
    print("-" * 60)

    # ----------------------------------------------------
    # 6. Evaluation Visualizations & Residual Analysis
    # ----------------------------------------------------
    print("\n[Step 6] Plotting Predictions and Residual Diagnostics...")
    p_act = plot_actual_vs_predicted(y_te, preds_dict, d_te, output_dir)
    print(f"  • Saved Actual vs Predicted plot: [outputs/actual_vs_predicted.png]({p_act})")

    primary_model_name = "Linear Regression"
    p_res = plot_residuals(y_te, preds_dict[primary_model_name], d_te, primary_model_name, output_dir)
    print(f"  • Saved Residual Analysis plot: [outputs/residuals.png]({p_res})")

    model_path = save_best_model(models_dict[primary_model_name], scaler, feat_cols, models_dir / "stock_forecasting_model.pkl")
    print(f"  • Saved Best Model Artifact: [models/stock_forecasting_model.pkl]({model_path})")

    # ----------------------------------------------------
    # 7. Future 10-Trading-Day Forecast
    # ----------------------------------------------------
    print("\n[Step 7] Generating Recursive 10-Trading-Day Future Forecast...")
    forecast_df, p_fc = recursive_future_forecast(
        model=models_dict[primary_model_name],
        scaler=scaler,
        feature_cols=feat_cols,
        df_clean=clean_df,
        num_future_days=10,
        output_dir=output_dir
    )

    print("\n" + "-" * 40)
    print("   10-DAY RECURSIVE FUTURE FORECAST TABLE")
    print("-" * 40)
    print(forecast_df.to_string(index=False))
    print("-" * 40)
    print(f"  • Saved Future Forecast plot: [outputs/future_forecast.png]({p_fc})")

    # ----------------------------------------------------
    # Financial Disclaimer
    # ----------------------------------------------------
    print("\n" + "=" * 75)
    print(" IMPORTANT FINANCIAL DISCLAIMER:")
    print(" This project is an educational time-series forecasting system based on")
    print(" historical market data. Predictions are statistical estimates and do")
    print(" NOT constitute financial advice, investment strategy, or guarantees")
    print(" of future stock market performance.")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
