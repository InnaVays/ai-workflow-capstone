import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_train_endpoint():
    response = requests.post(
        f"{BASE_URL}/train",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"data_dir": "data/cs-train", "test": True})
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Model training completed successfully"}

def test_predict_endpoint():
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"country": "all", "year": "2018", "month": "01", "day": "05", "test": True})
    )
    assert response.status_code == 200
    assert "y_pred" in response.json()

def test_logfile_endpoint():
    response = requests.post(
        f"{BASE_URL}/logfile",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"log_type": "train"})
    )
    assert response.status_code == 200
    assert "log_content" in response.json()
