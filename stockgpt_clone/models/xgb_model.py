"""XGBoost wrapper for training and persistence."""
from typing import Any
from xgboost import XGBRegressor
import joblib


def train_xgb(X, y, n_estimators=100, learning_rate=0.1) -> XGBRegressor:
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X, y)
    return model


def save_xgb_model(model: XGBRegressor, path: str) -> None:
    joblib.dump(model, path)


def load_xgb_model(path: str) -> Any:
    return joblib.load(path)

