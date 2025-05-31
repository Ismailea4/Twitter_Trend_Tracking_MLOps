import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import joblib
import os

def load_data(tweet_path, user_path):
    print("Loading data...")
    tweet_data = pd.read_csv(tweet_path)
    user_data = pd.read_csv(user_path)
    print(f"Loaded {len(tweet_data)} tweets and {len(user_data)} users.")
    return tweet_data, user_data

def load_bert_model(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    print("Loading BERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("BERT model loaded.")
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    text = str(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    sentiment = torch.argmax(probabilities, dim=1).item()
    return "POSITIVE" if sentiment == 1 else "NEGATIVE", probabilities[0][sentiment].item()

def add_bert_sentiment(df, tokenizer, model):
    if "bert_sentiment" not in df.columns or "bert_score" not in df.columns:
        print("Predicting sentiment for each text using BERT...")
        df["bert_sentiment"], df["bert_score"] = zip(*df["text"].apply(lambda x: predict_sentiment(x, tokenizer, model)))
        print("Sentiment prediction completed.")
    else:
        print("Sentiment columns already exist, skipping prediction.")
    return df

def encode_features(df):
    print("Encoding sentiment and company columns...")
    df['sentiment_numeric'] = df['bert_sentiment'].map({'NEGATIVE': -1, 'POSITIVE': 1})
    le = LabelEncoder()
    df['company_encoded'] = le.fit_transform(df['company'])
    print("Encoding completed.")
    return df

def add_lda_topics(df, n_topics=5):
    print(f"Performing LDA topic modeling with {n_topics} topics...")
    df['text'] = df['text'].astype(str)
    vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df['text'])
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_output = lda_model.fit_transform(doc_term_matrix)
    df['topic'] = np.argmax(lda_output, axis=1)
    for i in range(lda_output.shape[1]):
        df[f'topic_{i}'] = lda_output[:, i]
    print("LDA topic modeling completed.")
    return df

def segment_and_save_models(df, feature_cols, output_dir="segmentation_models"):
    print("Starting segmentation and model saving for each company...")
    os.makedirs(output_dir, exist_ok=True)
    company_groups = df.groupby('company')
    segmentation_results = {}

    for company, group in company_groups:
        print(f"\nProcessing company: {company}")
        X = group[feature_cols].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        best_score = -1
        best_k = 2
        best_kmeans = None

        print("Searching for optimal number of clusters (k) using silhouette score...")
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            print(f"  k={k}: silhouette_score={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)

        print(f"Best k for {company}: {best_k} (silhouette_score={best_score:.4f})")
        joblib.dump({'scaler': scaler, 'kmeans': best_kmeans}, f"{output_dir}/{company}_segmentation.pkl")
        segmentation_results[company] = {
            'best_k': best_k,
            'silhouette_score': best_score
        }

    print("\nSaving segmentation summary...")
    pd.DataFrame.from_dict(segmentation_results, orient='index').to_csv(f"{output_dir}/segmentation_summary.csv")
    print("Segmentation and model saving completed.")
    return segmentation_results

def main():
    tweet_data, user_data = load_data('../data/processed/cleaned_tweet_data.csv', '../data/processed/cleaned_user_data.csv')
    df = user_data.copy()
    tokenizer, model = load_bert_model()
    df = add_bert_sentiment(df, tokenizer, model)
    df = encode_features(df)
    df = add_lda_topics(df, n_topics=5)
    feature_cols = ['total_comments', 'total_replies', 'total_likes', 'sentiment_numeric', 'bert_score',
                    'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4']
    segment_and_save_models(df, feature_cols, output_dir="../segmentation_models")

if __name__ == "__main__":
    main()