"""Evaluation utilities: load artifacts and compute metrics on holdout test set."""
from pathlib import Path
import json
import yaml
import joblib
from .data import fetch_stock_data, add_technical_features, create_sequences, prepare_xgb_features
from .models.lstm_model import load_lstm_model
from .models.xgb_model import load_xgb_model
from .utils.metrics import rmse, mae
from .utils.logging_utils import configure_logger


def evaluate_from_config(config_path: str):
    logger = configure_logger()
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ticker = cfg['data']['ticker']
    start_date = cfg['data']['start_date']
    end_date = cfg['data']['end_date']
    look_back = cfg['data'].get('look_back', 60)
    artifact_dir = Path(cfg['artifacts']['dir'])

    # try to read metadata.json for artifact paths if present
    metadata_file = artifact_dir / 'metadata.json'
    metadata = None
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        logger.info(f"Loaded metadata from {metadata_file}")

    logger.info(f"Fetching data for evaluation: {ticker}")
    df = fetch_stock_data(ticker, start_date, end_date)
    df = add_technical_features(df)

    if cfg['model']['type'] == 'lstm':
        X, y, scaler = create_sequences(df['Close'].values, look_back=look_back)
        split = int(X.shape[0] * (1 - cfg['training']['test_size']))
        X_test, y_test = X[split:], y[split:]

        # discover model path via metadata if possible
        if metadata and 'model_path' in metadata:
            model_path = metadata['model_path']
        else:
            model_path = str(artifact_dir / 'lstm_model')
        logger.info(f"Loading LSTM model from: {model_path}")
        model = load_lstm_model(model_path)

        preds = model.predict(X_test)
        preds_rescaled = scaler.inverse_transform(preds)
        y_rescaled = scaler.inverse_transform(y_test)
        report = {'rmse': float(rmse(y_rescaled, preds_rescaled)), 'mae': float(mae(y_rescaled, preds_rescaled))}
        logger.info(f"Evaluation report: {report}")
        return report

    elif cfg['model']['type'] == 'xgboost':
        X_tab, y_tab = prepare_xgb_features(df, look_back=look_back)
        split = int(X_tab.shape[0] * (1 - cfg['training']['test_size']))
        X_test, y_test = X_tab[split:], y_tab[split:]

        # discover model path via metadata if possible
        if metadata and 'model_path' in metadata:
            model_path = metadata['model_path']
        else:
            model_path = str(artifact_dir / 'xgb.joblib')
        logger.info(f"Loading XGBoost model from: {model_path}")
        model = load_xgb_model(model_path)

        preds = model.predict(X_test)
        report = {'rmse': float(rmse(y_test, preds)), 'mae': float(mae(y_test, preds))}
        logger.info(f"Evaluation report: {report}")
        return report

    else:
        raise ValueError(f"Unsupported model type: {cfg['model']['type']}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()
    evaluate_from_config(args.config)
