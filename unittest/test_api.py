import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_train_endpoint():
    response = client.post("/train", json={"data_dir": "data/cs-train", "test": True})
    assert response.status_code == 200

def test_predict_endpoint():
    response = client.post("/predict", json={"country": "all", "year": "2018", "month": "01", "day": "05", "test": True})
    assert response.status_code == 200

if __name__ == "__main__":
    pytest.main()
