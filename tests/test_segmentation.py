import os
import joblib
import numpy as np
import pytest

@pytest.mark.order(4)
def test_segmentation_models_exist_and_predict():
    seg_dir = "segmentation_models"
    companies = [f.split("_")[0] for f in os.listdir(seg_dir) if f.endswith("_segmentation.pkl")]
    assert companies, "No segmentation models found in segmentation_models/"

    for company in companies:
        model_path = os.path.join(seg_dir, f"{company}_segmentation.pkl")
        model_bundle = joblib.load(model_path)
        scaler = model_bundle["scaler"]
        kmeans = model_bundle["kmeans"]

        # Create dummy data with the correct number of features (should match training)
        # Example: 10 samples, 10 features (adjust if your feature count is different)
        n_features = kmeans.cluster_centers_.shape[1]
        X = np.random.rand(5, n_features)
        X_scaled = scaler.transform(X)
        preds = kmeans.predict(X_scaled)
        assert preds.shape == (5,)

def test_segmentation_summary_exists():
    summary_path = os.path.join("segmentation_models", "segmentation_summary.csv")
    assert os.path.exists(summary_path), "Segmentation summary CSV not found"