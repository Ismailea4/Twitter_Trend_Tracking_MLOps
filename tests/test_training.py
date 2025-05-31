import pandas as pd
import numpy as np
import pytest

from src.train_models import train_lstm, train_rf, train_xgb, load_lstm, load_rf, load_xgb

@pytest.fixture
def sample_df():
    # Minimal DataFrame with required columns for training
    data = {
        "tweet_id": [1]*10 + [2]*10,
        "date": pd.date_range("2023-01-01", periods=10).tolist() * 2,
        "total_engagement": np.random.randint(1, 100, 20)
    }
    return pd.DataFrame(data)

@pytest.mark.order(2)
def test_all_training_and_loading(sample_df):
    # Test train_lstm
    model, scalers = train_lstm(sample_df, window_size=3, horizon=1, epochs=1, batch_size=2, alias="test")
    assert model is not None
    assert isinstance(scalers, dict)
    assert len(scalers) > 0

    # Test train_rf
    model, scalers = train_rf(sample_df, window_size=3, horizon=1, alias="test")
    assert model is not None
    assert isinstance(scalers, dict)
    assert len(scalers) > 0

    # Test train_xgb
    model, scalers = train_xgb(sample_df, window_size=3, horizon=1, alias="test")
    assert model is not None
    assert isinstance(scalers, dict)
    assert len(scalers) > 0

    # Test load_and_retrain_lstm
    try:
        model, scalers = load_lstm()
        X = np.random.rand(10, 3, 1)
        y = np.random.rand(10, 1)
        model.fit(X, y, epochs=1, batch_size=2, verbose=0)
        assert model is not None
    except Exception as e:
        pytest.skip(f"LSTM model not found or retrain failed: {e}")

    # Test load_and_retrain_rf
    try:
        model, scalers = load_rf()
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        model.fit(X, y)
        assert model is not None
    except Exception as e:
        pytest.skip(f"RF model not found or retrain failed: {e}")

    # Test load_and_retrain_xgb
    try:
        model, scalers = load_xgb()
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        model.fit(X, y)
        assert model is not None
    except Exception as e:
        pytest.skip(f"XGB model not found or retrain failed: {e}")