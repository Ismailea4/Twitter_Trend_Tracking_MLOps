import numpy as np
import pandas as pd
from datetime import timedelta


def forecast_lstm(tweet_id, df, model, scaler, window_size=6, days_ahead=5):
    tweet_df = df[df['tweet_id'] == tweet_id].sort_values('date')
    last_values = tweet_df['total_engagement'].values[-window_size:]
    if len(last_values) < window_size:
        raise ValueError("Not enough data to forecast for this tweet.")
    input_seq = scaler.transform(last_values.reshape(-1, 1)).reshape(1, window_size, 1)
    predictions = []
    for _ in range(days_ahead):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    start_date = tweet_df['date'].max() + timedelta(days=1)
    future_dates = [start_date + timedelta(days=i) for i in range(days_ahead)]
    return pd.DataFrame({'date': future_dates, 'forecast': predictions})

def forecast_rf(tweet_id, df, model, scaler, window_size=5, days_ahead=5):
    tweet_df = df[df['tweet_id'] == tweet_id].sort_values('date')
    last_values = tweet_df['total_engagement'].values[-window_size:]
    if len(last_values) < window_size:
        raise ValueError("Not enough data to forecast for this tweet.")
    input_seq = scaler.transform(last_values.reshape(-1, 1)).flatten()
    predictions = []
    for _ in range(days_ahead):
        pred_scaled = model.predict([input_seq])[0]
        predictions.append(pred_scaled)
        input_seq = np.append(input_seq[1:], pred_scaled)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    start_date = tweet_df['date'].max() + timedelta(days=1)
    future_dates = [start_date + timedelta(days=i) for i in range(days_ahead)]
    return pd.DataFrame({'date': future_dates, 'forecast': predictions})

def forecast_xgb(tweet_id, df, model, scaler, window_size=5, days_ahead=5):
    tweet_df = df[df['tweet_id'] == tweet_id].sort_values('date')
    last_values = tweet_df['total_engagement'].values[-window_size:]
    if len(last_values) < window_size:
        raise ValueError("Not enough data to forecast for this tweet.")
    input_seq = scaler.transform(last_values.reshape(-1, 1)).flatten()
    predictions = []
    for _ in range(days_ahead):
        pred_scaled = model.predict([input_seq])[0]
        predictions.append(pred_scaled)
        input_seq = np.append(input_seq[1:], pred_scaled)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    start_date = tweet_df['date'].max() + timedelta(days=1)
    future_dates = [start_date + timedelta(days=i) for i in range(days_ahead)]
    return pd.DataFrame({'date': future_dates, 'forecast': predictions})

def forecast_xgb_with_text(tweet_id, df, model, scaler, embedding_map, window_size=5, days_ahead=5):
    tweet_df = df[df["tweet_id"] == tweet_id].sort_values("date")
    last_values = tweet_df["total_engagement"].values[-window_size:]
    if len(last_values) < window_size:
        raise ValueError("Not enough data to forecast for this tweet.")
    input_seq = scaler.transform(last_values.reshape(-1, 1)).flatten()
    text_emb = embedding_map[tweet_id]
    predictions = []
    for _ in range(days_ahead):
        input_features = np.concatenate([input_seq, text_emb])
        pred_scaled = model.predict([input_features])[0]
        predictions.append(pred_scaled)
        input_seq = np.append(input_seq[1:], pred_scaled)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    start_date = tweet_df["date"].max() + timedelta(days=1)
    future_dates = [start_date + timedelta(days=i) for i in range(days_ahead)]
    return pd.DataFrame({"date": future_dates, "forecast": predictions})