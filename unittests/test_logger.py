import pytest
import os
import json
from model.logger import update_train_log, update_predict_log

LOG_DIR = "logs"

def test_update_train_log():
    tag = "test_tag"
    dates = ("2022-01-01", "2022-12-31")
    metrics = {"rmse": 100}
    runtime = "00:00:10"
    version = 0.1
    note = "test model"

    train_log_file = os.path.join(LOG_DIR, "train-test-log.json")
    if not os.path.exists(train_log_file):
        with open(train_log_file, 'w') as f:
            json.dump([], f)

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

    predict_log_file = os.path.join(LOG_DIR, "predict-test-log.json")
    if not os.path.exists(predict_log_file):
        with open(predict_log_file, 'w') as f:
            json.dump([], f)

    update_predict_log(country, y_pred, y_proba, target_date, runtime, version, test=True)

    with open(predict_log_file, "r") as file:
        logs = json.load(file)
        assert any(log["country"] == country for log in logs)

if __name__ == "__main__":
    pytest.main()