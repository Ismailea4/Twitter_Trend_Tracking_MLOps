from src.preprocess import remove_empty_files, preprocess_data
import os


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data_scraped")
    
    remove_empty_files(DATA_PATH, ['samsung', 'apple', 'nintendo'])
    df_tweets, df_users = preprocess_data()
    df_tweets.to_csv("cleaned_tweet_data.csv", index=False)
    df_users.to_csv("cleaned_user_data.csv", index=False)
    print("Preprocessing complete.")