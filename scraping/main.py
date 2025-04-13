# scraper/main.py
import os
import glob
from datetime import datetime
import pandas as pd
from scrape_post import scrape_twitter_comments2  # ou scrape_function.py

def get_urls_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def get_base_name(file_name):
    base = os.path.basename(file_name)
    name = base.replace("urls_", "").replace(".txt", "")
    return name

print("Scraping Twitter comments...")

if __name__ == "__main__":
    #txt_files = glob.glob("scraping/urls_*.txt")
    txt_files = ["scraping/urls_nintendo.txt"]
    today = datetime.today().strftime("%Y-%m-%d")

    for txt_file in txt_files:
        urls = get_urls_from_file(txt_file)
        category = get_base_name(txt_file)

        if urls:
            try:
                output_dir = f"../data_scraped/{category}/{today}/"
                os.makedirs(output_dir, exist_ok=True)
                scrape_twitter_comments2(urls, output_dir)

            except Exception as e:
                print(f"Error scraping {category}: {e}")
        else:
            print(f"No URLs found in {txt_file}")
