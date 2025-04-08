# scraper/main.py
import pandas as pd
from scrape_post import scrape_twitter_comments  # tu peux mettre ton code dans scrape_function.py

def get_last_url(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines[-1].strip() if lines else None

print("Scraping Twitter comments...")

if __name__ == "__main__":
    url = get_last_url("scraping/urls.txt")
    if url:
        df = scrape_twitter_comments(url)
        df.to_csv(f"scraping/data_scraped/output_{url.split('/')[-1]}.csv", index=False)
        print(f"Scraped data saved for: {url}")
    else:
        print("No URL found.")
