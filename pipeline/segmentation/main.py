import os
import pandas as pd
from src.segmentation import (
    load_data,
    load_bert_model,
    add_bert_sentiment,
    encode_features,
    add_lda_topics,
    segment_and_save_models
)

def run_segmentation(
    tweet_path="data/processed/cleaned_tweet_data.csv",
    user_path="data/processed/cleaned_user_data.csv",
    output_dir="segmentation_models",
    n_topics=5
):
    # Load data
    tweet_data, user_data = load_data(tweet_path, user_path)
    df = user_data.copy()

    # Sentiment analysis
    tokenizer, model = load_bert_model()
    df = add_bert_sentiment(df, tokenizer, model)
    df = encode_features(df)
    df = add_lda_topics(df, n_topics=n_topics)

    # Features for segmentation
    feature_cols = [
        'total_comments', 'total_replies', 'total_likes', 'sentiment_numeric', 'bert_score',
        'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4'
    ]
    segmentation_results = segment_and_save_models(df, feature_cols, output_dir=output_dir)

    # Assign cluster labels to each user (for each company)
    for company in df['company'].unique():
        model_bundle = pd.read_pickle(os.path.join(output_dir, f"{company}_segmentation.pkl"))
        company_mask = df['company'] == company
        X = df.loc[company_mask, feature_cols]
        X_scaled = model_bundle['scaler'].transform(X)
        df.loc[company_mask, 'segmentation_label'] = model_bundle['kmeans'].predict(X_scaled)

    return df

if __name__ == "__main__":
    df_segmented = run_segmentation()
    print(df_segmented.head())