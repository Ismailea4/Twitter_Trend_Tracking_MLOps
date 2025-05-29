import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

MODEL_DIR = "../models"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_lstm(df, window_size=6, horizon=1, epochs=10, batch_size=32):
    scalers = {}
    all_sequences = []
    tweet_ids = []
    for tweet_id, group in df.groupby('tweet_id'):
        group = group.sort_values('date')
        if len(group) < window_size + horizon:
            continue
        values = group['total_engagement'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values)
        scalers[tweet_id] = scaler
        for i in range(len(scaled_values) - window_size - horizon + 1):
            input_seq = scaled_values[i:i+window_size]
            target = scaled_values[i+window_size+horizon-1]
            all_sequences.append((input_seq, target))
            tweet_ids.append(tweet_id)
    X = np.array([seq[0] for seq in all_sequences])
    y = np.array([seq[1] for seq in all_sequences])
    model = Sequential()
    model.add(LSTM(64, input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    ensure_dir(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
    joblib.dump(scalers, os.path.join(MODEL_DIR, "lstm_scalers.pkl"))
    return model, scalers

def train_rf(df, window_size=5, horizon=1):
    X, y, tweet_ids, scalers = [], [], [], {}
    for tweet_id, group in df.groupby('tweet_id'):
        group = group.sort_values('date')
        if len(group) < window_size + horizon:
            continue
        scaler = MinMaxScaler()
        values = group['total_engagement'].values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        scalers[tweet_id] = scaler
        for i in range(len(scaled) - window_size - horizon + 1):
            X.append(scaled[i:i+window_size].flatten())
            y.append(scaled[i+window_size+horizon-1])
            tweet_ids.append(tweet_id)
    X = np.array(X)
    y = np.array(y)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y.ravel())
    ensure_dir(MODEL_DIR)
    joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(scalers, os.path.join(MODEL_DIR, "rf_scalers.pkl"))
    return model, scalers

def train_xgb(df, window_size=5, horizon=1):
    X, y, scalers = [], [], {}
    for tweet_id, group in df.groupby("tweet_id"):
        group = group.sort_values("date")
        if len(group) < window_size + horizon:
            continue
        scaler = MinMaxScaler()
        values = group["total_engagement"].values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        scalers[tweet_id] = scaler
        for i in range(len(scaled) - window_size - horizon + 1):
            X.append(scaled[i:i+window_size].flatten())
            y.append(scaled[i+window_size+horizon-1][0])
    X = np.array(X)
    y = np.array(y)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    ensure_dir(MODEL_DIR)
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    joblib.dump(scalers, os.path.join(MODEL_DIR, "xgb_scalers.pkl"))
    return model, scalers

def train_xgb_with_text(df, window_size=5, horizon=1, embedding_model_name='all-MiniLM-L6-v2'):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed.")
    df = df.copy()
    unique_tweets = df.drop_duplicates(subset="tweet_id")[["tweet_id", "text"]]
    text_model = SentenceTransformer(embedding_model_name)
    tweet_embeddings = text_model.encode(unique_tweets["text"].tolist(), show_progress_bar=True)
    embedding_map = {tweet_id: emb for tweet_id, emb in zip(unique_tweets["tweet_id"], tweet_embeddings)}
    X, y, scalers = [], [], {}
    for tweet_id, group in df.groupby("tweet_id"):
        group = group.sort_values("date")
        if len(group) < window_size + horizon:
            continue
        scaler = MinMaxScaler()
        values = group["total_engagement"].values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        scalers[tweet_id] = scaler
        text_emb = embedding_map[tweet_id]
        for i in range(len(scaled) - window_size - horizon + 1):
            time_features = scaled[i:i+window_size].flatten()
            combined_features = np.concatenate([time_features, text_emb])
            X.append(combined_features)
            y.append(scaled[i+window_size+horizon-1][0])
    X = np.array(X)
    y = np.array(y)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    ensure_dir(MODEL_DIR)
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_text_model.pkl"))
    joblib.dump(scalers, os.path.join(MODEL_DIR, "xgb_text_scalers.pkl"))
    joblib.dump(embedding_map, os.path.join(MODEL_DIR, "xgb_text_embedding_map.pkl"))
    return model, scalers, embedding_map

def load_lstm():
    model = load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))
    scalers = joblib.load(os.path.join(MODEL_DIR, "lstm_scalers.pkl"))
    return model, scalers

def load_rf():
    model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    scalers = joblib.load(os.path.join(MODEL_DIR, "rf_scalers.pkl"))
    return model, scalers

def load_xgb():
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    scalers = joblib.load(os.path.join(MODEL_DIR, "xgb_scalers.pkl"))
    return model, scalers

def load_xgb_with_text():
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_text_model.pkl"))
    scalers = joblib.load(os.path.join(MODEL_DIR, "xgb_text_scalers.pkl"))
    embedding_map = joblib.load(os.path.join(MODEL_DIR, "xgb_text_embedding_map.pkl"))
    return model, scalers, embedding_map

# Example usage:
if __name__ == "__main__":
    # Load your cleaned tweet data
    df = pd.read_csv("../processing/cleaned_tweet_data.csv", parse_dates=['date'])
    
    # Train models
    model, scalers = train_rf(df)
    model, scalers = train_xgb(df)
    model, scalers, embedding_map = train_xgb_with_text(df)
    model, scalers = train_lstm(df)
