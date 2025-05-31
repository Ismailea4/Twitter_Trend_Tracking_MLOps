import os
import sys
from src.preprocess import remove_empty_files, preprocess_data


if __name__ == "__main__":
    remove_empty_files("data/raw/", ['samsung', 'apple', 'nintendo'])
    df_tweets, df_users = preprocess_data("data/raw/", ['samsung', 'apple', 'nintendo'])
    df_tweets.to_csv("data/processed/cleaned_tweet_data.csv", index=False)
    df_users.to_csv("data/processed/cleaned_user_data.csv", index=False)
    print("Preprocessing complete.")