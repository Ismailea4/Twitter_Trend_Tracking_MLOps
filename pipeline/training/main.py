import os
import pandas as pd
from src.train_models import train_lstm, train_rf, train_xgb, train_xgb_with_text

def run_training(df=None, processed_path="data/processed/cleaned_tweet_data.csv"):
    """
    Train all forecasting models using the provided DataFrame or by loading from CSV.
    Args:
        df (pd.DataFrame or None): DataFrame to use for training. If None, loads from processed_path.
        processed_path (str): Path to the processed CSV file.
    """
    if df is None:
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found at {processed_path}")
        df = pd.read_csv(processed_path, parse_dates=['date'])
        print(f"Loaded data from {processed_path} with shape {df.shape}")
    else:
        print(f"Using provided DataFrame with shape {df.shape}")

    print("Training Random Forest...")
    rf_model, rf_scalers = train_rf(df)
    print("Training XGBoost...")
    xgb_model, xgb_scalers = train_xgb(df)
    print("Training XGBoost with Text Embeddings...")
    xgb_text_model, xgb_text_scalers, embedding_map = train_xgb_with_text(df)
    print("Training LSTM...")
    lstm_model, lstm_scalers = train_lstm(df)
    print("Training complete.")

    return {
        "rf": (rf_model, rf_scalers),
        "xgb": (xgb_model, xgb_scalers),
        "xgb_text": (xgb_text_model, xgb_text_scalers, embedding_map),
        "lstm": (lstm_model, lstm_scalers)
    }

if __name__ == "__main__":
    run_training()