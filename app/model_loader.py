import joblib


def load_model():
    return joblib.load("models/xgb_forecast_model.pkl")
