"""Simple FastAPI app to serve predictions from trained artifacts."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from stockgpt_clone.predict import predict_next_from_config

app = FastAPI(title="StockGPT Inference API")


class PredictRequest(BaseModel):
    config_path: str = 'config/default.yaml'


class PredictResponse(BaseModel):
    ticker: str
    next_price: float


@app.get('/health')
def health():
    return {"status": "ok"}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    cfg = req.config_path
    try:
        price = predict_next_from_config(cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # read ticker from yaml for response
    import yaml
    with open(cfg, 'r', encoding='utf-8') as f:
        conf = yaml.safe_load(f)
    return PredictResponse(ticker=conf['data']['ticker'], next_price=price)


if __name__ == '__main__':
    uvicorn.run('api.app:app', host='0.0.0.0', port=8000, reload=True)


