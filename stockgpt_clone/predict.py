"""Inference utilities for serving/predicting.

Provides simple functions that load artifacts and produce a 1-step ahead forecast.
"""
from pathlib import Path
import json
import yaml
import joblib
from .data import fetch_stock_data, add_technical_features
from .models.lstm_model import load_lstm_model
from .models.xgb_model import load_xgb_model
from .utils.logging_utils import configure_logger


def predict_next_from_config(config_path: str):
    logger = configure_logger()
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ticker = cfg['data']['ticker']
    start_date = cfg['data']['start_date']
    end_date = cfg['data']['end_date']
    look_back = cfg['data'].get('look_back', 60)
    artifact_dir = Path(cfg['artifacts']['dir'])

    logger.info(f"Fetching latest data for {ticker}")
    df = fetch_stock_data(ticker, start_date, end_date)
    df = add_technical_features(df)

    # try metadata.json to find artifact paths
    metadata_file = artifact_dir / 'metadata.json'
    metadata = None
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        logger.info(f"Loaded metadata from {metadata_file}")

    if cfg['model']['type'] == 'lstm':
        scaler_path = metadata.get('scaler_path') if metadata else str(artifact_dir / 'scaler.joblib')
        model_path = metadata.get('model_path') if metadata else str(artifact_dir / 'lstm_model')
        logger.info(f"Loading LSTM model from: {model_path}")
        logger.info(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        model = load_lstm_model(model_path)
        last_window = df['Close'].values[-look_back:]
        scaled = scaler.transform(last_window.reshape(-1, 1)).reshape(1, look_back, 1)
        pred_scaled = model.predict(scaled)
        pred = scaler.inverse_transform(pred_scaled)
        result = float(pred.ravel()[0])
        logger.info(f"LSTM next-step prediction: {result}")
        return result

    elif cfg['model']['type'] == 'xgboost':
        model_path = metadata.get('model_path') if metadata else str(artifact_dir / 'xgb.joblib')
        logger.info(f"Loading XGBoost model from: {model_path}")
        model = load_xgb_model(model_path)
        # prepare latest feature vector
        features = df[['Close', 'MA50', 'MA200']].iloc[-1].values.reshape(1, -1)
        pred = model.predict(features)
        result = float(pred.ravel()[0])
        logger.info(f"XGBoost next-step prediction: {result}")
        return result

    else:
        raise ValueError(f"Unsupported model type: {cfg['model']['type']}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()
    val = predict_next_from_config(args.config)
    print(f"Next step prediction: {val}")
