  # StockGPT Clone — Production-ready ML engineering template

This repository is a refactor of an educational Jupyter Notebook to a maintainable, testable, and deployable ML project for time-series stock price forecasting (task unchanged).

Checklist
- Audit the repository and list issues (Critical/High/Medium/Low) — done below
- Convert notebook to reusable modules (data, models, training, evaluation, prediction, utils) — implemented
- Add configuration, logging, error handling, and CLI — implemented
- Add FastAPI inference service and simple CLI — implemented
- Add requirements, architecture documentation, and improved README — implemented

What changed
- The monolithic notebook was converted into a small Python package `stockgpt_clone/` with clear separation of concerns: data, models, training, evaluation, and utilities.
- A `config/default.yaml` controls experiments and deployment settings.
- A FastAPI app (`api/app.py`) offers a lightweight prediction endpoint.

Quick start (development)
1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Train a model (default config):

```powershell
python cli.py train --config config/default.yaml
```

3. Run the API locally:

```powershell
uvicorn api.app:app --reload --port 8000
```

Repository layout (high level)

- stockgpt_clone/         # package with core logic
  - data.py               # data loading & preprocessing
  - models/               # model definitions & persistence
  - train.py              # training & artifact creation
  - evaluate.py           # evaluation scripts
  - predict.py            # inference helpers
  - utils/                # logging, metrics, IO helpers
- api/                    # FastAPI inference service
- config/                 # yaml configuration files
- cli.py                  # simple CLI wrapper
- requirements.txt        # pinned development dependencies
- ARCHITECTURE.md         # architecture + deployment notes

For details, see `ARCHITECTURE.md` and the package docstrings.
