import pytest
import os
import json
from fastapi.testclient import TestClient
from main import app
from model import model_train, model_load, model_predict, MODEL_DIR
from logger import update_train_log, update_predict_log

client = TestClient(app)

# Ensure the models directory exists
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# Ensure the logs directory exists
LOG_DIR = "logs"
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

# Ensure log files exist
train_log_file = os.path.join(LOG_DIR, "train_log.json")
predict_log_file = os.path.join(LOG_DIR, "predict_log.json")

if not os.path.exists(train_log_file):
    with open(train_log_file, 'w') as f:
        json.dump([], f)

if not os.path.exists(predict_log_file):
    with open(predict_log_file, 'w') as f:
        json.dump([], f)

# API tests
def test_train_endpoint():
    response = client.post("/train", json={"data_dir": "data/cs-train", "test": True})
    assert response.status_code == 200

def test_predict_endpoint():
    response = client.post("/predict", json={"country": "all", "year": "2018", "month": "01", "day": "05", "test": True})
    assert response.status_code == 200

# Model tests
def test_model_train():
    data_dir = "data/cs-train"
    model_train(data_dir, test=True)
    assert len(os.listdir(MODEL_DIR)) > 0, "No models were saved"

def test_model_load():
    all_data, all_models = model_load(prefix="test", data_dir="data/cs-train")
    assert all_data is not None
    assert all_models is not None

def test_model_predict():
    country = "united_kingdom"
    year = "2018"
    month = "01"
    day = "05"
    all_data, all_models = model_load(prefix="test", data_dir="data/cs-train")
    result = model_predict(country, year, month, day, all_models=all_models, test=True)
    assert "y_pred" in result

# Logging tests
def test_update_train_log():
    tag = "test_tag"
    dates = ("2022-01-01", "2022-12-31")
    metrics = {"rmse": 100}
    runtime = "00:00:10"
    version = 0.1
    note = "test model"

    update_train_log(tag, dates, metrics, runtime, version, note, test=True)
    
    with open(train_log_file, "r") as file:
        logs = json.load(file)
        assert any(log["tag"] == tag for log in logs)

def test_update_predict_log():
    country = "united_kingdom"
    y_pred = [100]
    y_proba = None
    target_date = "2023-01-01"
    runtime = "00:00:10"
    version = 0.1

    update_predict_log(country, y_pred, y_proba, target_date, runtime, version, test=True)

    with open(predict_log_file, "r") as file:
        logs = json.load(file)
        assert any(log["country"] == country for log in logs)

if __name__ == "__main__":
    pytest.main()
