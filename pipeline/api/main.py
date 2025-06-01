import os
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

from src.train_models import load_lstm, load_rf, load_xgb, load_xgb_with_text
from pipeline.prediction.main import run_prediction

app = FastAPI()

# Load models once at startup
lstm_model, lstm_scalers = load_lstm()
rf_model, rf_scalers = load_rf()
xgb_model, xgb_scalers = load_xgb()
xgb_text_model, xgb_text_scalers, embedding_map = load_xgb_with_text()

# Load tweet data once at startup
DATA_PATH = "data/processed/cleaned_tweet_data.csv"
df = pd.read_csv(DATA_PATH, parse_dates=['date'])

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>MLOps Model API</h2>
    <ul>
        <li><a href='/forecast-ui'>Engagement Forecasting UI</a></li>
        <li><a href='/segmentation-ui'>Customer Segmentation UI (coming soon)</a></li>
    </ul>
    """

@app.get("/forecast-ui", response_class=HTMLResponse)
def forecast_ui():
    tweet_ids = df['tweet_id'].unique()
    options = "".join([f"<option value='{tid}'>{tid}</option>" for tid in tweet_ids])
    return f"""
    <h2>Engagement Forecasting</h2>
    <form action="/forecast-result" method="get">
        <label for="tweet_id">Select Tweet ID:</label>
        <select name="tweet_id">{options}</select>
        <label for="days_ahead">Days Ahead:</label>
        <input type="number" name="days_ahead" value="5" min="1" max="30"/>
        <label for="model_type">Model:</label>
        <select name="model_type">
            <option value="lstm">LSTM</option>
            <option value="rf">Random Forest</option>
            <option value="xgb">XGBoost</option>
            <option value="xgb_text">XGBoost+Text</option>
        </select>
        <input type="submit" value="Forecast"/>
    </form>
    """

@app.get("/forecast-result", response_class=HTMLResponse)
def forecast_result(
    tweet_id: int = Query(...),
    days_ahead: int = Query(5),
    model_type: str = Query("lstm")
):
    # Get tweet info for latest date
    tweet_df = df[df['tweet_id'] == tweet_id].sort_values('date')
    if tweet_df.empty:
        return f"<h3>No tweet found for tweet_id {tweet_id}</h3>"

    latest = tweet_df.iloc[-1]
    username = latest['username']
    text = latest['text']
    date = latest['date'].strftime('%Y-%m-%d')
    replies = latest['replies']
    reposts = latest['reposts']
    likes = latest['likes']
    bookmarks = latest['bookmarks']
    views = latest['views']
    total_engagement = latest['total_engagement']

    # Forecast
    forecast_df = run_prediction(tweet_id, days_ahead=days_ahead, model_type=model_type)
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(tweet_df['date'], tweet_df['total_engagement'], label='Historical', marker='o')
    plt.plot(forecast_df['date'], forecast_df['forecast'], label='Forecast', linestyle='--', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Total Engagement')
    plt.title(f'Engagement Forecast for Tweet ID {tweet_id}')
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Render tweet-like rectangle
    tweet_html = f"""
    <div style='border:1px solid #ccc; border-radius:10px; width:400px; margin:20px auto; padding:16px; background:#f9f9f9;'>
        <div style='font-weight:bold; color:#1da1f2;'>{username}</div>
        <div style='margin:16px 0; font-size:1.1em;'>{text}</div>
        <div style='font-size:0.9em; color:#555;'>
            <span>Replies: {replies}</span> |
            <span>Reposts: {reposts}</span> |
            <span>Likes: {likes}</span> |
            <span>Bookmarks: {bookmarks}</span> |
            <span>Views: {views}</span> |
            <span>Date: {date}</span>
        </div>
        <div style='margin-top:8px; font-weight:bold;'>Total Engagement: {total_engagement}</div>
    </div>
    """

    html = f"""
    <h2>Engagement Forecasting Result</h2>
    <img src="data:image/png;base64,{img_base64}" style="display:block; margin:auto;"/>
    {tweet_html}
    <div style='text-align:center; margin-top:20px;'>
        <a href='/forecast-ui'>Back to Forecasting UI</a>
    </div>
    """
    return html