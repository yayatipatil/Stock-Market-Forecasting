"""
Future Forecasting Module

Implements recursive multi-step forecasting for future trading days (excluding weekends)
using trained model artifacts and historical context.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from feature_engineering import build_all_features, prepare_feature_matrix


def recursive_future_forecast(
    model: object,
    scaler: object,
    feature_cols: list[str],
    df_clean: pd.DataFrame,
    num_future_days: int = 10,
    output_dir: str | Path = None
) -> tuple[pd.DataFrame, str]:
    """
    Perform recursive multi-step forecasting for future business trading days.

    Parameters:
    -----------
    model : object
        Trained model instance.
    scaler : object
        Fitted StandardScaler.
    feature_cols : list[str]
        Ordered list of feature names.
    df_clean : pd.DataFrame
        Cleaned historical dataframe up to the latest known record.
    num_future_days : int
        Number of future trading days to forecast (default 10).
    output_dir : str or Path
        Directory to save forecast chart.

    Returns:
    --------
    tuple
        (forecast_df, chart_save_path)
    """
    # Working copy of historical data
    history_df = df_clean.copy()
    last_date = history_df["date"].max()

    # Generate future business days (Monday through Friday only)
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=num_future_days)

    forecast_records = []

    for future_date in future_dates:
        # Build features for current history
        feat_df = build_all_features(history_df)
        prep_df = prepare_feature_matrix(feat_df)

        # Get latest feature row (corresponds to predicting next step)
        latest_features = prep_df.iloc[-1][feature_cols].values.reshape(1, -1)

        # Scale features using training scaler
        latest_scaled = scaler.transform(latest_features)

        # Predict next closing price
        pred_close = model.predict(latest_scaled)[0]

        # Record prediction
        forecast_records.append({
            "date": future_date,
            "predicted_close": round(float(pred_close), 2)
        })

        # Append prediction into history to allow recursive lag calculation
        new_row = pd.DataFrame([{
            "date": future_date,
            "close": pred_close,
            "volume": history_df["volume"].iloc[-1], # placeholder
            "open": pred_close,
            "high": pred_close,
            "low": pred_close
        }])
        history_df = pd.concat([history_df, new_row], ignore_index=True)

    forecast_df = pd.DataFrame(forecast_records)

    save_path = ""
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "future_forecast.png"

        plt.figure(figsize=(12, 6))

        # Plot last 30 historical trading days
        recent_hist = df_clean.tail(30)
        plt.plot(recent_hist["date"], recent_hist["close"], label="Historical Close (Last 30 Days)", color="#1f77b4", linewidth=2.0, marker="o", markersize=3)

        # Plot future predictions
        plt.plot(forecast_df["date"], forecast_df["predicted_close"], label=f"Future {num_future_days}-Day Forecast", color="#d62728", linewidth=2.2, linestyle="--", marker="s", markersize=5)

        # Annotate forecast values
        for i, row in forecast_df.iterrows():
            plt.annotate(
                f"${row['predicted_close']:.2f}",
                (row["date"], row["predicted_close"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="#b2182b",
                fontweight="bold"
            )

        plt.title(f"Stock Price {num_future_days}-Trading-Day Recursive Future Forecast", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Date", fontsize=11, fontweight="bold")
        plt.ylabel("Predicted Close Price ($)", fontsize=11, fontweight="bold")
        plt.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        save_path = str(save_path)

    return forecast_df, save_path


if __name__ == "__main__":
    from data_preprocessing import load_raw_data, clean_and_sort_data

    root_dir = Path(__file__).resolve().parent.parent
    raw_df = load_raw_data(root_dir / "data" / "HistoricalQuotes.csv")
    clean_df = clean_and_sort_data(raw_df)

    # Load saved model artifact
    model_artifact = joblib.load(root_dir / "models" / "stock_forecasting_model.pkl")

    forecast_df, p_fc = recursive_future_forecast(
        model=model_artifact["model"],
        scaler=model_artifact["scaler"],
        feature_cols=model_artifact["feature_names"],
        df_clean=clean_df,
        num_future_days=10,
        output_dir=root_dir / "outputs"
    )

    print("--- 10-Day Future Forecast Table ---")
    print(forecast_df.to_string(index=False))
    print(f"Future forecast chart saved: {p_fc}")
