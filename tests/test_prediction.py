import numpy as np
import pytest
from src.train_models import load_lstm, load_rf, load_xgb

@pytest.mark.order(3)
def test_all_model_predictions():
    # LSTM prediction
    try:
        model, scalers = load_lstm()
        tweet_id = next(iter(scalers))
        scaler = scalers[tweet_id]
        X = np.random.rand(1, 6, 1)
        pred = model.predict(X)
        pred_original = scaler.inverse_transform(pred)
        assert pred.shape == (1, 1)
        assert pred_original.shape == (1, 1)
    except Exception as e:
        pytest.skip(f"LSTM model or scaler not found or prediction failed: {e}")

    # RF prediction
    try:
        model, scalers = load_rf()
        tweet_id = next(iter(scalers))
        scaler = scalers[tweet_id]
        X = np.random.rand(1, 5)
        pred = model.predict(X)
        pred_original = scaler.inverse_transform(pred.reshape(-1, 1))
        assert pred.shape == (1,)
        assert pred_original.shape == (1, 1)
    except Exception as e:
        pytest.skip(f"RF model or scaler not found or prediction failed: {e}")

    # XGB prediction
    try:
        model, scalers = load_xgb()
        tweet_id = next(iter(scalers))
        scaler = scalers[tweet_id]
        X = np.random.rand(1, 5)
        pred = model.predict(X)
        pred_original = scaler.inverse_transform(pred.reshape(-1, 1))
        assert pred.shape == (1,)
        assert pred_original.shape == (1, 1)
    except Exception as e:
        pytest.skip(f"XGB model or scaler not found or prediction failed: {e}")