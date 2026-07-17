"""Data loading and preprocessing utilities.

Functions:
 - fetch_stock_data: download OHLC data using yfinance
 - add_technical_features: add simple MAs
 - create_sequences: create LSTM sequences and labels
 - prepare_xgb_features: prepare tabular features for XGBoost
"""
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance.

    Returns cleaned DataFrame with Date index.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker} between {start_date} and {end_date}")
    df = df.dropna()
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df = df.dropna()
    return df


def create_sequences(values: np.ndarray, look_back: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Create LSTM sequences from 1-D price array.

    Returns (X, y, scaler) where X has shape (n_samples, look_back, 1).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values.reshape(-1, 1))
    sequences = []
    labels = []
    for i in range(len(scaled) - look_back):
        sequences.append(scaled[i:i + look_back])
        labels.append(scaled[i + look_back])
    X = np.array(sequences)
    y = np.array(labels).reshape(-1, 1)
    return X, y, scaler


def prepare_xgb_features(df: pd.DataFrame, look_back: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and target for XGBoost using Close, MA50, MA200.

    Aligns with LSTM sequence trimming by dropping the first `look_back` rows.
    """
    df2 = df[['Close', 'MA50', 'MA200']].iloc[look_back:].copy()
    target = df['Close'].values[look_back:]
    return df2.values, target

