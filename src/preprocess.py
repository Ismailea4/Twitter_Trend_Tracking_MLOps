import os
import re
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"#\w+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def remove_empty_files(base_dir, companies):
    for company in companies:
        company_path = os.path.join(base_dir, company)
        for date_folder in os.listdir(company_path):
            date_path = os.path.join(company_path, date_folder)
            for csv_file in os.listdir(date_path):
                if csv_file.endswith(".csv"):
                    file_path = os.path.join(date_path, csv_file)
                    if os.stat(file_path).st_size <= 2:
                        os.remove(file_path)
                        print(f"Removed empty file: {file_path}")

def preprocess_data(base_dir="../data_scraped/", 
                    companies=['samsung', 'apple', 'nintendo'], 
                    companies_username=['SamsungMobile','theapplehub','NintendoAmerica']):

    tweet_records = []
    user_records = {}

    for company in companies:
        company_path = os.path.join(base_dir, company)
        
        for date_folder in tqdm(os.listdir(company_path), desc=f"Processing {company}"):
            date_path = os.path.join(company_path, date_folder)
            if not os.path.exists(date_path) or not os.path.isdir(date_path):
                print(f"Directory does not exist or is not a folder: {date_path}")
                continue # Skip if the path does not exist or is not a directory
            
            for csv_file in os.listdir(date_path):
                if not csv_file.endswith(".csv"):
                    continue

                file_path = os.path.join(date_path, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    if df.empty:
                        continue
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                main = df.iloc[0]
                comments = df.iloc[1:]

                if main['username'] in companies_username:
                    main_sentiment = TextBlob(str(main['text'])).sentiment.polarity
                    comment_sentiments = comments['text'].dropna().apply(lambda x: TextBlob(str(x)).sentiment.polarity)
                    avg_comment_sentiment = comment_sentiments.mean() if not comment_sentiments.empty else None

                    tweet_records.append({
                        "tweet_id": csv_file.replace(".csv", ""),
                        "company": company,
                        "date": date_folder,
                        "username": main['username'],
                        "text": clean_text(main['text']),
                        "replies": main['replies'],
                        "reposts": main['reposts'],
                        "likes": main['likes'],
                        "bookmarks": main['bookmarks'],
                        "views": main['views'],
                        "comment_count": len(comments),
                        "total_engagement": main['replies'] + main['likes'] + main['bookmarks'] + main['reposts'],
                        "main_sentiment": main_sentiment,
                        "avg_comment_sentiment": avg_comment_sentiment
                    })

                for _, row in comments.iterrows():
                    username = row['username']
                    if pd.isna(username):
                        continue

                    sentiment = TextBlob(str(row['text'])).sentiment.polarity
                    if username not in user_records:
                        user_records[username] = {
                            "company": company,
                            "username": username,
                            "text": '',
                            "total_comments": 0,
                            "total_replies": 0,
                            "total_likes": 0,
                            "sentiments": []
                        }

                    user_records[username]["text"] += str(row['text']) + " "
                    user_records[username]["total_comments"] += 1
                    user_records[username]["total_replies"] += row.get("replies", 0)
                    user_records[username]["total_likes"] += row.get("likes", 0)
                    user_records[username]["sentiments"].append(sentiment)

    df_tweets = pd.DataFrame(tweet_records)
    df_users = pd.DataFrame([
        {
            **data,
            "text": clean_text(data["text"]),
            "avg_sentiment": sum(data["sentiments"]) / len(data["sentiments"]) if data["sentiments"] else 0
        } 
        for user, data in user_records.items()
    ])

    return df_tweets, df_users

if __name__ == "__main__":
    remove_empty_files("../data_scraped/", ['samsung', 'apple', 'nintendo'])
    df_tweets, df_users = preprocess_data()
    #df_tweets.to_csv("cleaned_tweet_data.csv", index=False)
    #df_users.to_csv("cleaned_user_data.csv", index=False)
    print("Preprocessing complete.")
