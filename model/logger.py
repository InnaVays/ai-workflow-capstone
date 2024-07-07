import os
import json
from datetime import datetime
import numpy as np 

LOG_DIR = "logs"
TRAIN_LOG_FILE = "train_log.json"
PREDICT_LOG_FILE = "predict_log.json"

def _write_log(log_data, log_file):
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    log_path = os.path.join(LOG_DIR, log_file)

    if os.path.exists(log_path):
        with open(log_path, "r") as file:
            logs = json.load(file)
    else:
        logs = []

    logs.append(log_data)

    with open(log_path, "w") as file:
        json.dump(logs, file, indent=4)

def update_train_log(tag, dates, metrics, runtime, model_version, model_version_note, test=False):
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "dates": dates,
        "metrics": metrics,
        "runtime": runtime,
        "model_version": model_version,
        "model_version_note": model_version_note,
        "test": test
    }
    _write_log(log_data, TRAIN_LOG_FILE)
    print(f"Training log updated for {tag}")

def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=False, business_metric=None):
    # Ensure business_metric is a dictionary
    if not isinstance(business_metric, dict):
        business_metric = {
            "absolute_error": None,
            "mse": None,
            "r2_score": None
        }
    
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "country": country,
        "y_pred": y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
        "y_proba": y_proba.tolist() if isinstance(y_proba, np.ndarray) else y_proba,
        "target_date": target_date,
        "runtime": runtime,
        "model_version": model_version,
        "test": test,
        "business_metric": business_metric
    }
    _write_log(log_data, PREDICT_LOG_FILE)
    print(f"Prediction log updated for {country} on {target_date}")