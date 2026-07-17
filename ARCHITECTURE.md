# Architecture and design notes

Overview
--------
This repository converts a single Jupyter notebook into a small ML application with the following goals:

- Clear separation of concerns: data ingestion, preprocessing, model training, evaluation, and inference.
- Reproducible configuration via YAML.
- Simple artifact management (models and scalers saved to `artifacts/`).
- Lightweight inference API using FastAPI for serving predictions.

Components
----------
- `stockgpt_clone/data.py` - data download and preprocessing. Keeps data-related logic isolated and testable.
- `stockgpt_clone/models` - model factories and save/load helpers for LSTM (TensorFlow) and XGBoost.
- `stockgpt_clone/train.py` - training entrypoint which reads a YAML config and emits artifacts.
- `stockgpt_clone/evaluate.py` - loads artifacts and computes evaluation metrics on a holdout set.
- `stockgpt_clone/predict.py` - loads artifacts and produces a 1-step ahead forecast; used by the API.
- `api/app.py` - FastAPI app that wraps prediction for simple deployment.
- `config/default.yaml` - example configuration used to reproduce experiments.

Deployment
----------
For low-traffic inference, run the FastAPI app with Uvicorn:

```powershell
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

For production, containerize the application and use an ASGI server with a process manager and autoscaling behind a load balancer.

Notes and next steps
--------------------
- Add unit tests for data transforms and model save/load.
- Integrate experiment tracking (MLflow or Weights & Biases).
- Add CI that lints, runs unit tests, and validates packaging.
- Add model versioning and schema validation for API payloads.

