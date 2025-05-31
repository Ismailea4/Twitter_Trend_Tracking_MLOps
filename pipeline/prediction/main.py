import os
import pandas as pd
from src.train_models import load_lstm, load_rf, load_xgb, load_xgb_with_text
from src.prediction import forecast_lstm, forecast_rf, forecast_xgb, forecast_xgb_with_text



def run_prediction(
    tweet_id,
    days_ahead=5,
    processed_path="data/processed/cleaned_tweet_data.csv",
    model_type="rf"
):
    """
    Forecast future engagement for a tweet_id using the specified model.
    Args:
        tweet_id: The tweet ID to forecast.
        days_ahead: Number of days to forecast.
        processed_path: Path to processed CSV.
        model_type: One of "lstm", "rf", "xgb", "xgb_text".
    Returns:
        forecast_df: DataFrame with forecasted dates and values.
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found at {processed_path}")
    df = pd.read_csv(processed_path, parse_dates=['date'])

    if model_type == "lstm":
        lstm_model, lstm_scalers = load_lstm()
        scaler = lstm_scalers[tweet_id]
        forecast_df = forecast_lstm(tweet_id, df, lstm_model, scaler, days_ahead=days_ahead)
    elif model_type == "rf":
        rf_model, rf_scalers = load_rf()
        scaler = rf_scalers[tweet_id]
        forecast_df = forecast_rf(tweet_id, df, rf_model, scaler, days_ahead=days_ahead)
    elif model_type == "xgb":
        xgb_model, xgb_scalers = load_xgb()
        scaler = xgb_scalers[tweet_id]
        forecast_df = forecast_xgb(tweet_id, df, xgb_model, scaler, days_ahead=days_ahead)
    elif model_type == "xgb_text":
        xgb_text_model, xgb_text_scalers, embedding_map = load_xgb_with_text()
        scaler = xgb_text_scalers[tweet_id]
        forecast_df = forecast_xgb_with_text(tweet_id, df, xgb_text_model, scaler, embedding_map, days_ahead=days_ahead)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return forecast_df

if __name__ == "__main__":
    # Example usage: forecast for the tweet_id with the most data using RF
    df = pd.read_csv("data/processed/cleaned_tweet_data.csv", parse_dates=['date'])
    tweet_id = df['tweet_id'].value_counts().idxmax()
    forecast = run_prediction(tweet_id, days_ahead=10, model_type="rf")
    print(forecast)