from pipeline.scraping.main import run_scraping
from pipeline.processing.main import run_processing
from pipeline.training.main import run_training
from pipeline.segmentation.main import run_segmentation

def main():
    print("=== Scraping ===")
    run_scraping(email=None, pseudo=None, password=None)  # Replace with actual credentials if needed
    print("Scraping completed. Data saved to data/raw/")
    print("=== Preprocessing ===")
    df_tweets, df_users = run_processing()
    print("=== Training Forecasting Models ===")
    run_training(df=df_tweets)
    print("=== User Segmentation ===")
    run_segmentation()
    print("Pipeline complete.")

if __name__ == "__main__":
    main()