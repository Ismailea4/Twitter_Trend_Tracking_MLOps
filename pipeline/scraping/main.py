# scraper/main.py
import os
import glob
from datetime import datetime
from src.scrape_post import scrape_twitter_comments2

def get_urls_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def get_base_name(file_name):
    base = os.path.basename(file_name)
    name = base.replace("urls_", "").replace(".txt", "")
    return name

def run_scraping(url_files=None, email=None, pseudo=None, password=None):
    """
    Scrape Twitter comments from all url files or a specific list.
    Args:
        url_files (list or None): List of url .txt files to scrape. If None, scrape all in scraping/urls_*.txt
    """
    print("Scraping Twitter comments...")
    if url_files is None:
        url_files = glob.glob("scraping/urls_*.txt")
    today = datetime.today().strftime("%Y-%m-%d")

    for txt_file in url_files:
        urls = get_urls_from_file(txt_file)
        category = get_base_name(txt_file)

        if urls:
            try:
                output_dir = f"data/raw/{category}/{today}/"
                print(f"Scraping {category} to {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                scrape_twitter_comments2(urls, output_dir, email, pseudo, password)
            except Exception as e:
                print(f"Error scraping {category}: {e}")
        else:
            print(f"No URLs found in {txt_file}")

if __name__ == "__main__":
    run_scraping()
