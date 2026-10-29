import os
import joblib
import pandas as pd
import pytest

@pytest.fixture
def model_and_data():
    model = joblib.load("models/model.joblib")
    current = pd.read_csv("data/current.csv")
    return model, current

def test_model_file_exists():
    """Verify the trained model file exists on disk."""
    assert os.path.exists("models/model.joblib"), "Model file missing!"

def test_model_predictions(model_and_data):
    """Verify model output shape and expected class labels."""
    model, current = model_and_data
    X_curr = current.drop(columns=["target"])
    preds = model.predict(X_curr)
    
    assert len(preds) == len(current), "Prediction count mismatch!"
    assert set(preds).issubset({0, 1, 2}), "Invalid prediction classes found!"