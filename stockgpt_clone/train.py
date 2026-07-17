"""Training entrypoint for StockGPT models.

This module reads a YAML config, trains either an LSTM or XGBoost model,
and writes artifacts (models, scaler, metadata) to disk.
"""
from pathlib import Path
import yaml
import joblib
from .data import fetch_stock_data, add_technical_features, create_sequences, prepare_xgb_features
from .models.lstm_model import build_lstm_model, save_lstm_model
from .models.xgb_model import train_xgb, save_xgb_model
from .utils.logging_utils import configure_logger
from .utils.io import save_json
from datetime import datetime


def train_from_config(config_path: str) -> None:
    logger = configure_logger()
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ticker = cfg['data']['ticker']
    start_date = cfg['data']['start_date']
    end_date = cfg['data']['end_date']
    look_back = cfg['data'].get('look_back', 60)
    artifact_dir = Path(cfg['artifacts']['dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    df = fetch_stock_data(ticker, start_date, end_date)
    df = add_technical_features(df)

    if cfg['model']['type'] == 'lstm':
        logger.info("Preparing sequences for LSTM")
        X, y, scaler = create_sequences(df['Close'].values, look_back=look_back)
        # simple train/test split chronological
        split = int(X.shape[0] * (1 - cfg['training']['test_size']))
        X_train, y_train = X[:split], y[:split]
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        logger.info("Starting LSTM training")
        model.fit(X_train, y_train, epochs=cfg['training']['epochs'], batch_size=cfg['training']['batch_size'], validation_split=cfg['training']['val_split'])
        model_path_base = artifact_dir / 'lstm_model'
        saved_model_path = save_lstm_model(model, str(model_path_base))
        scaler_path = str(artifact_dir / 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved LSTM model to {saved_model_path} and scaler to {scaler_path}")
        # write metadata
        metadata = {
            'model_type': 'lstm',
            'model_path': saved_model_path,
            'scaler_path': scaler_path,
            'config_path': config_path,
            'trained_at': datetime.utcnow().isoformat() + 'Z'
        }
        save_json(str(artifact_dir / 'metadata.json'), metadata)
        logger.info(f"Wrote metadata to {artifact_dir / 'metadata.json'}")

    elif cfg['model']['type'] == 'xgboost':
        logger.info("Preparing tabular features for XGBoost")
        X_tab, y_tab = prepare_xgb_features(df, look_back=look_back)
        split = int(X_tab.shape[0] * (1 - cfg['training']['test_size']))
        X_train, y_train = X_tab[:split], y_tab[:split]
        model = train_xgb(X_train, y_train, n_estimators=cfg['model'].get('n_estimators', 100), learning_rate=cfg['model'].get('learning_rate', 0.1))
        xgb_path = str(artifact_dir / 'xgb.joblib')
        save_xgb_model(model, xgb_path)
        logger.info(f"Saved XGBoost model to {xgb_path}")
        metadata = {
            'model_type': 'xgboost',
            'model_path': xgb_path,
            'config_path': config_path,
            'trained_at': datetime.utcnow().isoformat() + 'Z'
        }
        save_json(str(artifact_dir / 'metadata.json'), metadata)
        logger.info(f"Wrote metadata to {artifact_dir / 'metadata.json'}")

    else:
        raise ValueError(f"Unsupported model type: {cfg['model']['type']}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()
    train_from_config(args.config)


