"""
Time-Series Analysis Module

Handles weekly/monthly resampling, exploratory visualizations, trend analysis
with moving averages, and statsmodels time-series decomposition.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Set default aesthetic plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 1.0


def resample_data(df: pd.DataFrame) -> dict:
    """
    Resample daily stock data to Weekly and Monthly frequencies
    and compute aggregated metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe with 'date' column.

    Returns:
    --------
    dict
        Dictionary containing 'weekly' and 'monthly' DataFrames.
    """
    df_temp = df.set_index("date")

    # Resample Weekly (ending Sunday/Monday)
    weekly = df_temp.resample("W").agg({
        "close": ["mean", "max", "min", "std"],
        "volume": ["mean", "sum"]
    })
    weekly.columns = ["close_mean", "close_max", "close_min", "close_std", "volume_mean", "volume_sum"]
    weekly = weekly.dropna().reset_index()

    # Resample Monthly (Month end)
    monthly = df_temp.resample("ME" if hasattr(pd, "date_range") else "M").agg({
        "close": ["mean", "max", "min", "std"],
        "volume": ["mean", "sum"]
    })
    monthly.columns = ["close_mean", "close_max", "close_min", "close_std", "volume_mean", "volume_sum"]
    monthly = monthly.dropna().reset_index()

    return {"weekly": weekly, "monthly": monthly}


def plot_eda(df: pd.DataFrame, output_dir: str | Path) -> str:
    """
    Generate Exploratory Data Analysis plots: Date vs Close, Volume, Price Distribution.

    Saved to outputs/stock_price.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "stock_price.png"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False, gridspec_kw={"height_ratios": [2, 1, 1]})

    # 1. Closing Price over time
    axes[0].plot(df["date"], df["close"], color="#1f77b4", linewidth=2.0, label="Close Price ($)")
    axes[0].set_title("Stock Historical Closing Price (Aug 2018 - Aug 2019)", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_ylabel("Closing Price ($)", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # 2. Volume over time
    axes[1].bar(df["date"], df["volume"] / 1e6, color="#2ca02c", alpha=0.7, width=1.5, label="Volume (Millions)")
    axes[1].set_title("Daily Trading Volume", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Volume (M)", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # 3. Price Distribution
    sns.histplot(df["close"], kde=True, color="#ff7f0e", ax=axes[2], bins=30)
    axes[2].set_title("Distribution of Closing Prices", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Closing Price ($)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Frequency", fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return str(save_path)


def calculate_moving_averages(df: pd.DataFrame, output_dir: str | Path) -> tuple[pd.DataFrame, str]:
    """
    Calculate 5-day, 10-day, and 20-day moving averages and generate trend plot.

    Saved to outputs/moving_average.png
    """
    df_ma = df.copy()
    df_ma["ma_5"] = df_ma["close"].rolling(window=5).mean()
    df_ma["ma_10"] = df_ma["close"].rolling(window=10).mean()
    df_ma["ma_20"] = df_ma["close"].rolling(window=20).mean()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "moving_average.png"

    plt.figure(figsize=(12, 6))
    plt.plot(df_ma["date"], df_ma["close"], label="Actual Close", color="#333333", alpha=0.6, linewidth=1.5)
    plt.plot(df_ma["date"], df_ma["ma_5"], label="5-Day MA (Short-term)", color="#e377c2", linewidth=1.8)
    plt.plot(df_ma["date"], df_ma["ma_10"], label="10-Day MA (Medium-term)", color="#17becf", linewidth=1.8)
    plt.plot(df_ma["date"], df_ma["ma_20"], label="20-Day MA (Monthly Trend)", color="#d62728", linewidth=2.0)

    plt.title("Stock Price Trend & Moving Averages (5, 10, 20 Days)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11, fontweight="bold")
    plt.ylabel("Price ($)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return df_ma, str(save_path)


def decompose_series(df: pd.DataFrame, period: int = 20, output_dir: str | Path = None) -> tuple[object, str]:
    """
    Perform additive statsmodels time-series decomposition.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing 'date' and 'close'.
    period : int
        Decomposition period (default: 20 trading days ~ 1 month).
    output_dir : str or Path
        Directory to save outputs/decomposition.png.

    Returns:
    --------
    tuple (DecomposeResult, str)
        Decomposition result object and plot save path.
    """
    df_ts = df.set_index("date")["close"]

    # Use forward fill for any date gaps if necessary
    decomposition = seasonal_decompose(df_ts, model="additive", period=period, extrapolate_trend="freq")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "decomposition.png"

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(decomposition.observed, color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("Observed", fontsize=10, fontweight="bold")
    axes[0].set_title(f"Time-Series Additive Decomposition (Period = {period} Trading Days)", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(decomposition.trend, color="#ff7f0e", linewidth=2.0)
    axes[1].set_ylabel("Trend", fontsize=10, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(decomposition.seasonal, color="#2ca02c", linewidth=1.2)
    axes[2].set_ylabel("Seasonality", fontsize=10, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    axes[3].scatter(decomposition.resid.index, decomposition.resid, color="#d62728", s=12, alpha=0.7)
    axes[3].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("Residuals", fontsize=10, fontweight="bold")
    axes[3].set_xlabel("Date", fontsize=11, fontweight="bold")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return decomposition, str(save_path)


if __name__ == "__main__":
    from data_preprocessing import load_raw_data, clean_and_sort_data

    root_dir = Path(__file__).resolve().parent.parent
    raw_df = load_raw_data(root_dir / "data" / "HistoricalQuotes.csv")
    clean_df = clean_and_sort_data(raw_df)
    out_dir = root_dir / "outputs"

    res = resample_data(clean_df)
    print("--- Resampling Test ---")
    print(f"Weekly periods: {len(res['weekly'])}, Monthly periods: {len(res['monthly'])}")

    p1 = plot_eda(clean_df, out_dir)
    print(f"EDA plot saved: {p1}")

    df_ma, p2 = calculate_moving_averages(clean_df, out_dir)
    print(f"MA plot saved: {p2}")

    decomp, p3 = decompose_series(clean_df, period=20, output_dir=out_dir)
    print(f"Decomposition plot saved: {p3}")
