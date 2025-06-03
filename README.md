---

````markdown
# Twitter_Trend_Tracking_MLOps

This project implements an end-to-end MLOps pipeline to forecast tweet engagement and segment Twitter users based on their interactions with selected tech companies (Apple, Samsung, Nintendo). It integrates machine learning, data scraping, natural language processing, deployment, and monitoring into a reproducible and scalable system.

## 📌 Features

- Scrapes tweets and comments using Selenium.
- Predicts tweet engagement using models like XGBoost and LSTM.
- Segments users with KMeans clustering and LDA topic modeling.
- Serves results via a FastAPI backend and a Streamlit dashboard.
- Modularized pipeline with Docker containerization and CI/CD.

## 🚀 How to Run the Pipeline

1. **Edit Twitter Credentials**  
   Before running, open `pipeline_all.py` and add your:

   - **Email**
   - **Username (pseudo)**
   - **Password**  
     These credentials are required to log in to Twitter and scrape data.

2. **Configure Scraping URLs**

   - Go to the `pipeline/scraping/` folder.
   - Add or edit the `.txt` files containing the tweet URLs you want to scrape (one URL per line).

3. **Run the Full Pipeline**
   ```bash
   python pipeline_all.py
   ```
````

## 🖥️ Launch the App

You can run the Streamlit dashboard directly to explore forecasts and user segments:

```bash
streamlit run streamlit_app.py
```

## 🧰 Tech Stack

- **Languages**: Python
- **Libraries**: Selenium, BeautifulSoup, pandas, scikit-learn, XGBoost, MLflow, Streamlit, FastAPI
- **MLOps Tools**: Docker, GitHub Actions, DVC, MLflow
- **Deployment**: Render.com

## 📁 Project Structure

```
├── pipeline/
│   ├── scraping/          # Tweet data scraping
│   ├── processing/        # Preprocessing & feature engineering
│   ├── training/          # Model training scripts
│   ├── segmentation/      # User clustering & topic modeling
│   ├── prediction/        # Forecasting logic
│   ├── api/               # FastAPI backend
├── streamlit_app.py       # Streamlit frontend
├── pipeline_all.py        # Full pipeline execution script
├── requirements.txt
├── docker-compose.yml
```

## 📄 License

This project is developed as part of a final-year AI engineering project at **ENSIAS** and is not intended for commercial use.

## 🙏 Acknowledgements

Special thanks to **M. Mohamed Lazaar** (My professor) and **M. Abdellatif El Afia** (Head of AI Sector) at ENSIAS for their supervision and support throughout this project.
