import os
from fastapi import FastAPI, Query
import pandas as pd
from src.train_models import load_lstm, load_rf, load_xgb, load_xgb_with_text
from pipeline.prediction.main import run_prediction

app = FastAPI()

# Load models once at startup
lstm_model, lstm_scalers = load_lstm()
rf_model, rf_scalers = load_rf()
xgb_model, xgb_scalers = load_xgb()
xgb_text_model, xgb_text_scalers, embedding_map = load_xgb_with_text()

@app.get("/")
def root():
    return {"message": "MLOps Model API is running."}

@app.get("/forecast/")
def forecast(
    tweet_id: int = Query(..., description="Tweet ID to forecast"),
    days_ahead: int = Query(5, description="Number of days to forecast"),
    model_type: str = Query("lstm", description="Model type: lstm, rf, xgb, xgb_text")
):
    try:
        forecast_df = run_prediction(
            tweet_id=tweet_id,
            days_ahead=days_ahead,
            model_type=model_type
        )
        # Convert DataFrame to list of dicts for JSON response
        return {"forecast": forecast_df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}