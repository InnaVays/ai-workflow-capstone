import pytest
import os
from model.model import model_train, model_load, model_predict, MODEL_DIR

def test_model_train():
    data_dir = "data/cs-train"
    model_train(data_dir, test=True)
    assert len(os.listdir(MODEL_DIR)) > 0, "No models were saved"

def test_model_load():
    all_data, all_models = model_load(prefix="test", data_dir="data/cs-train")
    assert all_data is not None
    assert all_models is not None

def test_model_predict():
    data_dir = "data/cs-train"
    model_train(data_dir, test=True)
    
    country = "united_kingdom"
    year = "2018"
    month = "01"
    day = "05"

    try:
        all_data, all_models = model_load(prefix="test", data_dir="data/cs-train")
        assert all_data is not None, "all_data is None"
        assert all_models is not None, "all_models is None"
    except Exception as e:
        assert False, f"Failed to load models or data: {str(e)}"

    try:
        result = model_predict(country, year, month, day, all_models=all_models, test=True)
        assert "y_pred" in result
    except Exception as e:
        assert False, f"Failed to predict: {str(e)}"

if __name__ == "__main__":
    pytest.main()