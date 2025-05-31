import os
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import joblib

import mlflow
import mlflow.xgboost

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

MODEL_DIR = "models"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_lstm(df, window_size=6, horizon=2, epochs=10, batch_size=32,alias="challenger"):
    with mlflow.start_run(run_name="LSTM"):
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

        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)

        # Log
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("input_shape", X.shape[1:])
        mlflow.log_param("output_shape", y.shape[1:])
        mlflow.log_param("loss_function", "mse")
        mlflow.log_param("optimizer", "adam")
        # Log metrics
        score = model.evaluate(X, y, verbose=0)
        mlflow.log_metric("train_score", score)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        # Log model and register it
        mlflow.tensorflow.autolog()
        mlflow.keras.log_model(model, artifact_path="model", registered_model_name="LSTMModel")
        # Optional: Register alias 'challenger'
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(name="LSTMModel", stages=["None"])[0].version
        client.set_registered_model_alias(name="LSTMModel", alias=alias, version=latest_version)
        

        ensure_dir(MODEL_DIR)
        model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
        joblib.dump(scalers, os.path.join(MODEL_DIR, "lstm_scalers.pkl"))
        return model, scalers

def train_rf(df, window_size=5, horizon=1,alias="challenger"):
    with mlflow.start_run(run_name="RandomForest"):
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

        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)

        # Logging
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("n_estimators", 100)
        
        # Log metrics
        score = model.score(X, y)
        mlflow.log_metric("train_score", score)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        # Log model and register it
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="RandomForestModel")
        # Optional: Register alias 'challenger'
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(name="RandomForestModel", stages=["None"])[0].version
        client.set_registered_model_alias(name="RandomForestModel", alias=alias, version=latest_version)
        
        ensure_dir(MODEL_DIR)
        joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.pkl"))
        joblib.dump(scalers, os.path.join(MODEL_DIR, "rf_scalers.pkl"))
        return model, scalers

def train_xgb(df, window_size=5, horizon=1,alias="challenger"):
    with mlflow.start_run(run_name="XGBoost"):
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
        
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)

        # Logging
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("horizon", horizon)
        # Log metrics
        score = model.score(X, y)
        mlflow.log_metric("train_score", score)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        # Log model and register it
        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name="XGBoostModel")
        # Optional: Register alias 'challenger'
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(name="XGBoostModel", stages=["None"])[0].version
        client.set_registered_model_alias(name="XGBoostModel", alias=alias, version=latest_version)
                
        ensure_dir(MODEL_DIR)
        joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
        joblib.dump(scalers, os.path.join(MODEL_DIR, "xgb_scalers.pkl"))
        return model, scalers

def train_xgb_with_text(df, window_size=5, horizon=1, embedding_model_name='all-MiniLM-L6-v2', model_name="xgb_text_model",alias="challenger"):
    with mlflow.start_run(run_name="XGBoost with Text Embeddings"):
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
        
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)
        
        # Log parameters
        mlflow.log_param("model_type", "XGBoost with Text Embeddings")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("horizon", horizon)
        # Log metrics (example: training score)
        score = model.score(X, y)
        mlflow.log_metric("train_score", score)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        # Log model and register it
        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=model_name)

        # Optional: Register alias 'challenger'
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version
        client.set_registered_model_alias(name=model_name, alias=alias, version=latest_version)

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
    mlflow.set_experiment("twitter_engagement_forecasting")
    # Load your cleaned tweet data
    df = pd.read_csv("data/processed/cleaned_tweet_data.csv", parse_dates=['date'])
    
    # Train models
    model, scalers = train_rf(df)
    model, scalers = train_xgb(df)
    model, scalers, embedding_map = train_xgb_with_text(df)
    model, scalers = train_lstm(df)
