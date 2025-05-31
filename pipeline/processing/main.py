import os
from src.preprocess import remove_empty_files, preprocess_data


def run_processing(raw_dir="data/raw/", processed_dir="data/processed/", companies=None):
    if companies is None:
        companies = ['samsung', 'apple', 'nintendo']
    remove_empty_files(raw_dir, companies)
    df_tweets, df_users = preprocess_data(raw_dir, companies)
    os.makedirs(processed_dir, exist_ok=True)
    df_tweets.to_csv(os.path.join(processed_dir, "cleaned_tweet_data.csv"), index=False)
    df_users.to_csv(os.path.join(processed_dir, "cleaned_user_data.csv"), index=False)
    print("Preprocessing complete.")
    return df_tweets, df_users


if __name__ == "__main__":
    run_processing()