"""LSTM model builder and persistence utilities."""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def save_lstm_model(model: tf.keras.Model, path: str) -> str:
    """Save a Keras model ensuring a proper file extension.

    If the provided path has no `.keras` or `.h5` extension, `.keras` will be appended.
    Returns the final path used for saving.
    """
    path_str = str(path)
    if not path_str.endswith('.keras') and not path_str.endswith('.h5'):
        path_str = path_str + '.keras'
    model.save(path_str)
    return path_str


def load_lstm_model(path: str) -> tf.keras.Model:
    """Load a Keras model. If the exact path doesn't exist, try common extensions.

    Accepts either a file with extension or a base path (e.g., 'artifacts/lstm_model').
    """
    import os

    path_str = str(path)
    # direct load if exists
    if os.path.exists(path_str):
        return load_model(path_str)
    # try common extensions
    for ext in ('.keras', '.h5'):
        candidate = path_str + ext
        if os.path.exists(candidate):
            return load_model(candidate)
    # fallback: let load_model raise the appropriate error
    return load_model(path_str)

