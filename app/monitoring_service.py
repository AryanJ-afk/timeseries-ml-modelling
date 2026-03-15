import os
import pandas as pd

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_24",
    "lag_48",
    "rolling_mean_24",
    "rolling_std_24",
    "hour",
    "day_of_week",
    "month"
]

REFERENCE_PATH = "data/processed/train_features.csv"
INFERENCE_HISTORY_PATH = "data/processed/inference_history.csv"
MIN_ROWS_FOR_DRIFT = 500

reference_df = pd.read_csv(REFERENCE_PATH, index_col=0, parse_dates=True)

REF_MEANS = reference_df[FEATURE_COLUMNS].mean()
REF_STDS = reference_df[FEATURE_COLUMNS].std().replace(0, 1e-6)


def validate_feature_columns(df: pd.DataFrame):
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in FEATURE_COLUMNS]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected extra columns: {extra}")


def append_to_history(df: pd.DataFrame) -> int:
    os.makedirs("data/processed", exist_ok=True)

    if os.path.exists(INFERENCE_HISTORY_PATH):
        history_df = pd.read_csv(INFERENCE_HISTORY_PATH)
        updated_df = pd.concat([history_df, df], ignore_index=True)
    else:
        updated_df = df.copy()

    updated_df.to_csv(INFERENCE_HISTORY_PATH, index=False)
    return len(updated_df)


def compute_feature_drift_score(recent_df: pd.DataFrame) -> dict:
    numeric_features = [
        "lag_1",
        "lag_2",
        "lag_24",
        "lag_48",
        "rolling_mean_24",
        "rolling_std_24",
    ]

    n = len(recent_df)

    recent_means = recent_df[numeric_features].mean()
    recent_stds = recent_df[numeric_features].std().replace(0, 1e-6)

    ref_means = REF_MEANS[numeric_features]
    ref_stds = REF_STDS[numeric_features].replace(0, 1e-6)

    # Use standard error for batch mean comparison
    mean_se = ref_stds / (n ** 0.5)
    mean_z = ((recent_means - ref_means) / mean_se).abs()

    # Compare std changes more sensitively
    std_ratio = (recent_stds / ref_stds).replace([float("inf"), -float("inf")], 0).fillna(0)
    std_shift = (std_ratio - 1.0).abs()

    mean_drifted_features = int((mean_z > 3).sum())
    std_drifted_features = int((std_shift > 0.25).sum())

    total_drifted_features = len(set(
        list(mean_z[mean_z > 3].index) + list(std_shift[std_shift > 0.25].index)
    ))

    drift_score = float(mean_z.mean() + std_shift.mean())

    drift_detected = total_drifted_features >= 2

    drift_details = {
        "mean_z_scores": mean_z.round(4).to_dict(),
        "std_shift_scores": std_shift.round(4).to_dict(),
        "mean_drifted_features": mean_drifted_features,
        "std_drifted_features": std_drifted_features,
        "total_drifted_features": total_drifted_features,
    }

    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "details": drift_details
    }



def run_feature_drift_detection() -> dict:
    if not os.path.exists(INFERENCE_HISTORY_PATH):
        return {
            "drift_checked": False,
            "drift_detected": False,
            "drift_score": 0.0,
            "message": "No stored inference history yet.",
            "details": {}
        }

    history_df = pd.read_csv(INFERENCE_HISTORY_PATH)

    if len(history_df) < MIN_ROWS_FOR_DRIFT:
        return {
            "drift_checked": False,
            "drift_detected": False,
            "drift_score": 0.0,
            "message": f"Stored inputs below {MIN_ROWS_FOR_DRIFT}. Feature drift not checked yet.",
            "details": {}
        }

    recent_df = history_df.tail(MIN_ROWS_FOR_DRIFT)

    drift_result = compute_feature_drift_score(recent_df)

    if drift_result["drift_detected"]:
        message = (
            "Feature drift detected in recent inputs. Predictions are still generated, "
            "but retraining is recommended."
        )
    else:
        message = "No significant feature drift detected in recent inputs."

    return {
        "drift_checked": True,
        "drift_detected": bool(drift_result["drift_detected"]),
        "drift_score": float(drift_result["drift_score"]),
        "message": message,
        "details": drift_result["details"]
    }
