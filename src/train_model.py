import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os


def create_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})

    # lag features
    df["lag_1"] = df["y"].shift(1)
    df["lag_2"] = df["y"].shift(2)
    df["lag_24"] = df["y"].shift(24)
    df["lag_48"] = df["y"].shift(48)

    # rolling features
    df["rolling_mean_24"] = df["y"].shift(1).rolling(24).mean()
    df["rolling_std_24"] = df["y"].shift(1).rolling(24).std()

    # time-based features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    df = df.dropna()
    return df


def train_forecasting_model(path: str):
    series = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")

    df = create_features(series)

    # time-based split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df.drop(columns=["y"])
    y_train = train_df["y"]

    X_test = test_df.drop(columns=["y"])
    y_test = test_df["y"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # save artifacts
    joblib.dump(model, "models/xgb_forecast_model.pkl")
    train_df.to_csv("data/processed/train_features.csv")
    test_df.to_csv("data/processed/test_features.csv")

    # save baseline metrics
    with open("models/baseline_metrics.txt", "w") as f:
        f.write(f"MAE={mae:.6f}\n")
        f.write(f"RMSE={rmse:.6f}\n")

    return model


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_forecasting_model("data/processed/hourly_demand.csv")