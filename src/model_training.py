"""
Model Training and Evaluation Module

Handles chronological train/test splitting, fitting standard scaling without leakage,
training Naive Baseline, Linear Regression, Random Forest, and Gradient Boosting models,
computing evaluation metrics (MAE, RMSE, R², MAPE), residual analysis, saving models,
and generating visual plots.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate MAE, RMSE, R², and MAPE performance metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth actual values.
    y_pred : np.ndarray
        Model predicted values.

    Returns:
    --------
    dict
        Dictionary containing metric values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE (%)": round(mape, 4)
    }


def chronological_split(
    df_prepared: pd.DataFrame,
    target_col: str = "close",
    train_ratio: float = 0.8
) -> tuple:
    """
    Perform a strict chronological split of time-series data without shuffling.
    Scales features using StandardScaler fit ONLY on training data to prevent leakage.

    Parameters:
    -----------
    df_prepared : pd.DataFrame
        Prepared feature dataframe containing 'date', target, and feature columns.
    target_col : str
        Target column name (default 'close').
    train_ratio : float
        Proportion of data for training (default 0.8).

    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, dates_train, dates_test, scaler, feature_cols)
    """
    non_feature_cols = ["date", "open", "high", "low", "volume", target_col]
    feature_cols = [col for col in df_prepared.columns if col not in non_feature_cols]

    split_idx = int(len(df_prepared) * train_ratio)

    df_train = df_prepared.iloc[:split_idx].copy()
    df_test = df_prepared.iloc[split_idx:].copy()

    dates_train = df_train["date"].values
    dates_test = df_test["date"].values

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, dates_train, dates_test, scaler, feature_cols


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    df_prepared: pd.DataFrame,
    train_ratio: float = 0.8
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Train Naive Baseline, Linear Regression, Random Forest, and Gradient Boosting.
    Evaluate all models on unseen test data.

    Returns:
    --------
    tuple
        (summary_results_df, trained_models_dict, predictions_dict)
    """
    split_idx = int(len(df_prepared) * train_ratio)
    df_test = df_prepared.iloc[split_idx:].copy()

    # 1. Naive Baseline: Tomorrow's Close = Today's Close (lag_1)
    naive_pred = df_test["lag_1"].values

    # 2. Linear Regression (Primary Model)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    # 3. Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # 4. Gradient Boosting Regressor
    gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbr_model.fit(X_train, y_train)
    gbr_pred = gbr_model.predict(X_test)

    models_dict = {
        "Linear Regression": lr_model,
        "Random Forest": rf_model,
        "Gradient Boosting": gbr_model
    }

    predictions_dict = {
        "Naive Baseline": naive_pred,
        "Linear Regression": lr_pred,
        "Random Forest": rf_pred,
        "Gradient Boosting": gbr_pred
    }

    # Assemble performance evaluation table
    results_list = []
    for name, pred in predictions_dict.items():
        metrics = calculate_metrics(y_test, pred)
        metrics["Model"] = name
        results_list.append(metrics)

    results_df = pd.DataFrame(results_list)[["Model", "MAE", "RMSE", "R2", "MAPE (%)"]]

    return results_df, models_dict, predictions_dict


def plot_actual_vs_predicted(
    y_test: np.ndarray,
    predictions_dict: dict,
    dates_test: np.ndarray,
    output_dir: str | Path
) -> str:
    """
    Generate graph comparing Actual Close vs Predicted Close across test set.

    Saved to outputs/actual_vs_predicted.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "actual_vs_predicted.png"

    plt.figure(figsize=(12, 6))

    # Convert dates to pandas Datetime for clean plotting if needed
    dates = pd.to_datetime(dates_test)

    plt.plot(dates, y_test, label="Actual Close", color="#111111", linewidth=2.5, marker="o", markersize=3)
    plt.plot(dates, predictions_dict["Naive Baseline"], label="Naive Baseline", color="#ff7f0e", linestyle="--", linewidth=1.5, alpha=0.8)
    plt.plot(dates, predictions_dict["Linear Regression"], label="Linear Regression (Primary)", color="#1f77b4", linewidth=2.0)

    if "Random Forest" in predictions_dict:
        plt.plot(dates, predictions_dict["Random Forest"], label="Random Forest", color="#2ca02c", linestyle=":", linewidth=1.5)
    if "Gradient Boosting" in predictions_dict:
        plt.plot(dates, predictions_dict["Gradient Boosting"], label="Gradient Boosting", color="#d62728", linestyle="-.", linewidth=1.5)

    plt.title("Actual vs Predicted Stock Closing Price (Test Data)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11, fontweight="bold")
    plt.ylabel("Closing Price ($)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return str(save_path)


def plot_residuals(
    y_test: np.ndarray,
    best_pred: np.ndarray,
    dates_test: np.ndarray,
    model_name: str,
    output_dir: str | Path
) -> str:
    """
    Calculate residuals (Actual - Predicted) and create residual plot and distribution.

    Saved to outputs/residuals.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "residuals.png"

    residuals = y_test - best_pred
    dates = pd.to_datetime(dates_test)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1.5, 1]})

    # 1. Residuals over Time
    axes[0].plot(dates, residuals, color="#d62728", marker="o", markersize=4, linestyle="-", linewidth=1.2, label=f"Residuals ({model_name})")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"Residual Analysis (Actual - Predicted) for {model_name}", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_ylabel("Residual Error ($)", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # 2. Residual Distribution
    sns.histplot(residuals, kde=True, color="#9467bd", ax=axes[1], bins=20)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_title("Distribution of Prediction Residuals", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residual Error ($)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Frequency", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return str(save_path)


def save_best_model(
    model: object,
    scaler: object,
    feature_cols: list[str],
    save_path: str | Path
) -> str:
    """
    Save trained model, scaler, and feature specifications via joblib.

    Parameters:
    -----------
    model : object
        Trained model instance.
    scaler : object
        Fitted StandardScaler.
    feature_cols : list[str]
        Ordered list of feature column names.
    save_path : str or Path
        Destination filepath.

    Returns:
    --------
    str
        Path to saved file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_cols
    }

    joblib.dump(artifact, save_path)
    return str(save_path)


if __name__ == "__main__":
    from data_preprocessing import load_raw_data, clean_and_sort_data
    from feature_engineering import build_all_features, prepare_feature_matrix

    root_dir = Path(__file__).resolve().parent.parent
    raw_df = load_raw_data(root_dir / "data" / "HistoricalQuotes.csv")
    clean_df = clean_and_sort_data(raw_df)
    feat_df = build_all_features(clean_df)
    prep_df = prepare_feature_matrix(feat_df)

    X_train_s, X_test_s, y_train, y_test, d_train, d_test, scaler, f_cols = chronological_split(prep_df)

    results_df, models_dict, preds_dict = train_and_evaluate_models(X_train_s, X_test_s, y_train, y_test, prep_df)

    print("--- Model Evaluation Table ---")
    print(results_df.to_string(index=False))

    out_dir = root_dir / "outputs"
    p_act = plot_actual_vs_predicted(y_test, preds_dict, d_test, out_dir)
    print(f"Actual vs Predicted chart saved: {p_act}")

    p_res = plot_residuals(y_test, preds_dict["Linear Regression"], d_test, "Linear Regression", out_dir)
    print(f"Residuals chart saved: {p_res}")

    model_path = save_best_model(models_dict["Linear Regression"], scaler, f_cols, root_dir / "models" / "stock_forecasting_model.pkl")
    print(f"Trained model saved: {model_path}")
