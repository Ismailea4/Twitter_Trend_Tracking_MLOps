import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
import base64
import re
import os

# Set your FastAPI backend URL (update if deployed elsewhere)
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Twitter Trend Tracking", layout="centered")
st.title("🐦 Twitter Trend Tracking Dashboard")

# --- Engagement Forecasting Section ---
st.header("📈 Engagement Forecasting")

@st.cache_data
def load_tweet_data():
    df = pd.read_csv("data/processed/cleaned_tweet_data.csv", parse_dates=["date"])
    return df

tweet_df = load_tweet_data()
tweet_ids = tweet_df['tweet_id'].unique()
tweet_id = st.selectbox("Select Tweet ID", tweet_ids)

# Show tweet details for selected tweet_id
selected_tweet = tweet_df[tweet_df['tweet_id'] == tweet_id].sort_values('date').iloc[-1]
with st.container():
    st.markdown(
        f"""
        <div style='border:1px solid #ccc; border-radius:10px; width:100%; margin:10px 0; padding:16px; background:#f9f9f9;'>
            <div style='font-weight:bold; color:#1da1f2; font-size:1.1em;'>{selected_tweet['username']}</div>
            <div style='margin:12px 0; font-size:1.1em;'>{selected_tweet['text']}</div>
            <div style='font-size:0.95em; color:#555;'>
                <span>Replies: {selected_tweet['replies']}</span> |
                <span>Reposts: {selected_tweet['reposts']}</span> |
                <span>Likes: {selected_tweet['likes']}</span> |
                <span>Bookmarks: {selected_tweet['bookmarks']}</span> |
                <span>Views: {selected_tweet['views']}</span> |
                <span>Date: {selected_tweet['date'].strftime('%Y-%m-%d')}</span>
            </div>
            <div style='margin-top:8px; font-weight:bold;'>Total Engagement: {selected_tweet['total_engagement']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

days_ahead = st.slider("Days Ahead", 1, 30, 5)
model_type = st.selectbox("Model", ["lstm", "rf", "xgb", "xgb_text"], format_func=lambda x: {
    "lstm": "LSTM",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "xgb_text": "XGBoost + Text"
}[x])

if st.button("🔮 Forecast Engagement"):
    params = {
        "tweet_id": tweet_id,
        "days_ahead": days_ahead,
        "model_type": model_type
    }
    with st.spinner("Fetching forecast..."):
        response = requests.get(f"{API_URL}/forecast-result", params=params)
        if response.status_code == 200:
            html = response.text
            # Extract base64 image
            img_match = re.search(r'src="data:image/png;base64,([^"]+)"', html)
            if img_match:
                img_data = base64.b64decode(img_match.group(1))
                st.image(Image.open(io.BytesIO(img_data)), caption="Forecast Plot", use_column_width=True)
            # Extract tweet info card
            tweet_html = re.search(r"(<div style='border:1px[^>]+>.+?</div>)", html, re.DOTALL)
            if tweet_html:
                st.markdown(tweet_html.group(1), unsafe_allow_html=True)
            # Show navigation link
            st.markdown("<div style='text-align:center; margin-top:20px;'><a href='#engagement-forecasting'>Back to Forecasting UI</a></div>", unsafe_allow_html=True)
        else:
            st.error("Error fetching forecast result.")

st.markdown("---")

# --- Customer Segmentation Section ---
st.header("👥 Customer Segmentation")

@st.cache_data
def load_user_data():
    df = pd.read_csv("data/processed/cleaned_user_data.csv")
    return df

user_df = load_user_data()
companies = user_df['company'].unique()
company = st.selectbox("Select Company", companies)

if st.button("📊 Show Segmentation"):
    params = {"company": company}
    with st.spinner("Running segmentation..."):
        response = requests.get(f"{API_URL}/segmentation-result", params=params)
        if response.status_code == 200:
            html = response.text
            # Extract base64 image
            img_match = re.search(r'src="data:image/png;base64,([^"]+)"', html)
            if img_match:
                img_data = base64.b64decode(img_match.group(1))
                st.image(Image.open(io.BytesIO(img_data)), caption="Segmentation Pie Chart", use_column_width=True)
            # Extract segment info cards
            segments = re.findall(r"(<div style='border:1px[^>]+>.+?</div>)", html, re.DOTALL)
            for seg_html in segments:
                st.markdown(seg_html, unsafe_allow_html=True)
            # Show navigation link
            st.markdown("<div style='text-align:center; margin-top:20px;'><a href='#customer-segmentation'>Back to Segmentation UI</a></div>", unsafe_allow_html=True)
        else:
            st.error("Error fetching segmentation result.")

st.markdown(
    """
    <style>
    .stButton>button {background-color: #1da1f2; color: white;}
    </style>
    """,
    unsafe_allow_html=True
)