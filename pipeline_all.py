import os
import sys
import glob
import pandas as pd

# 1. Scraping
from scraping.main import get_urls_from_file, get_base_name
from src.scrape_post import scrape_twitter_comments2

# 2. Preprocessing
from src.preprocess import remove_empty_files, preprocess_data

# 3. Forecasting Models
from src.train_models import train_lstm, train_rf, train_xgb, train_xgb_with_text

# 4. Segmentation
from src.segmentation import main as segmentation_main

def run_scraping():
    print("=== Scraping ===")
    txt_files = glob.glob("scraping/urls_*.txt")
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    for txt_file in txt_files:
        urls = get_urls_from_file(txt_file)
        category = get_base_name(txt_file)
        if urls:
            output_dir = f"data/raw/{category}/{today}/"
            os.makedirs(output_dir, exist_ok=True)
            scrape_twitter_comments2(urls, output_dir)

def run_preprocessing():
    print("=== Preprocessing ===")
    remove_empty_files("data/raw/", ['samsung', 'apple', 'nintendo'])
    df_tweets, df_users = preprocess_data("data/raw/", ['samsung', 'apple', 'nintendo'])
    os.makedirs("data/processed", exist_ok=True)
    df_tweets.to_csv("data/processed/cleaned_tweet_data.csv", index=False)
    df_users.to_csv("data/processed/cleaned_user_data.csv", index=False)
    return df_tweets, df_users

def run_forecasting(df_tweets):
    print("=== Training Forecasting Models ===")
    train_rf(df_tweets)
    train_xgb(df_tweets)
    train_xgb_with_text(df_tweets)
    train_lstm(df_tweets)

def run_segmentation():
    print("=== User Segmentation ===")
    segmentation_main()

def main():
    run_scraping()
    df_tweets, df_users = run_preprocessing()
    run_forecasting(df_tweets)
    run_segmentation()
    print("Pipeline complete.")

if __name__ == "__main__":
    main()