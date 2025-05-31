import os
import pytest
from src.preprocess import preprocess_data

@pytest.mark.order(1)
def test_preprocess_data_runs_and_outputs():
    # Use the default arguments or specify a test directory if you have one
    df_tweets, df_users = preprocess_data(
        base_dir="data/raw/",
        companies=['samsung', 'apple', 'nintendo'],
        companies_username=['SamsungMobile','theapplehub','NintendoAmerica']
    )

    # Check that DataFrames are not empty
    assert not df_tweets.empty, "df_tweets is empty"
    assert not df_users.empty, "df_users is empty"

    # Check for expected columns in tweets
    expected_tweet_cols = {
        "tweet_id", "company", "date", "username", "text", "replies", "reposts",
        "likes", "bookmarks", "views", "comment_count", "total_engagement",
        "main_sentiment", "avg_comment_sentiment"
    }
    assert expected_tweet_cols.issubset(df_tweets.columns), "Missing columns in df_tweets"

    # Check for expected columns in users
    expected_user_cols = {
        "company", "username", "text", "total_comments", "total_replies",
        "total_likes", "avg_sentiment"
    }
    assert expected_user_cols.issubset(df_users.columns), "Missing columns in df_users"