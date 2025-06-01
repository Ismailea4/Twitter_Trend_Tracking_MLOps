import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from src.train_models import load_lstm, load_rf, load_xgb, load_xgb_with_text
from pipeline.prediction.main import run_prediction
from pipeline.segmentation.main import run_segmentation

app = FastAPI()

# Load models and data at startup
lstm_model, lstm_scalers = load_lstm()
rf_model, rf_scalers = load_rf()
xgb_model, xgb_scalers = load_xgb()
xgb_text_model, xgb_text_scalers, embedding_map = load_xgb_with_text()

DATA_PATH = "data/processed/cleaned_tweet_data.csv"
df = pd.read_csv(DATA_PATH, parse_dates=['date'])

USER_DATA_PATH = "data/processed/cleaned_user_data.csv"
user_df = pd.read_csv(USER_DATA_PATH)

@app.get("/forecast-result", response_class=HTMLResponse)
def forecast_result(
    tweet_id: int = Query(...),
    days_ahead: int = Query(5),
    model_type: str = Query("lstm")
):
    tweet_df = df[df['tweet_id'] == tweet_id].sort_values('date')
    if tweet_df.empty:
        return "<h3>No tweet found for tweet_id {}</h3>".format(tweet_id)

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

    forecast_df = run_prediction(tweet_id, days_ahead=days_ahead, model_type=model_type)
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
    <img src="data:image/png;base64,{img_base64}" style="display:block; margin:auto;"/>
    {tweet_html}
    """
    return html

@app.get("/segmentation-result", response_class=HTMLResponse)
def segmentation_result(company: str = Query(...)):
    try:
        df_segmented = run_segmentation(
            tweet_path="data/processed/cleaned_tweet_data.csv",
            user_path="data/processed/cleaned_user_data.csv",
            output_dir="segmentation_models",
            n_topics=5
        )
    except Exception as e:
        return f"<h3>Error running segmentation pipeline: {str(e)}</h3>"

    company_users = df_segmented[df_segmented['company'] == company]
    if company_users.empty:
        return f"<h3>No users found for company {company}</h3>"

    if 'segmentation_label' not in company_users.columns:
        return "<h3>Segmentation label not found. Please check the segmentation pipeline.</h3>"

    segment_counts = company_users["segmentation_label"].value_counts().sort_index()
    plt.figure(figsize=(6, 6))
    plt.pie(segment_counts, labels=[f"Segment {i}" for i in segment_counts.index], autopct='%1.1f%%', startangle=140, colors=plt.cm.tab10.colors)
    plt.title(f"User Segments for {company.capitalize()}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    html_segments = ""
    for seg in sorted(company_users["segmentation_label"].unique()):
        seg_users = company_users[company_users["segmentation_label"] == seg]
        usernames = ", ".join(seg_users["username"].astype(str).tolist())
        avg_comments = seg_users["total_comments"].mean()
        avg_replies = seg_users["total_replies"].mean()
        avg_likes = seg_users["total_likes"].mean()
        avg_sentiment = seg_users["avg_sentiment"].mean() if "avg_sentiment" in seg_users.columns else "-"
        html_segments += f"""
        <div style='border:1px solid #ccc; border-radius:8px; margin:16px; padding:12px;'>
            <b>Segment {seg}</b><br>
            <b>Usernames:</b> {usernames}<br>
            <b>Avg Comments:</b> {avg_comments:.2f} |
            <b>Avg Replies:</b> {avg_replies:.2f} |
            <b>Avg Likes:</b> {avg_likes:.2f} |
            <b>Avg Sentiment:</b> {avg_sentiment if avg_sentiment == '-' else f"{avg_sentiment:.2f}"}
        </div>
        """

    html = f"""
    <img src="data:image/png;base64,{img_base64}" style="display:block; margin:auto;"/>
    {html_segments}
    """
    return html