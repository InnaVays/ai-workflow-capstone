import os
import json
from datetime import datetime
import numpy as np

LOG_DIR = "logs"
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

def update_train_log(tag, dates, metrics, runtime, model_version, model_version_note, test=False):
    """
    update_train_log(tag, dates, metrics, runtime, model_version, model_version_note, test=False)
    
    Update train log file
    """
    log_entry = {
        "timestamp": str(datetime.now()), 
        "tag": tag, 
        "dates": dates, 
        "metrics": metrics,
        "runtime": runtime, 
        "model_version": model_version, 
        "model_version_note": model_version_note
    }

    log_file = os.path.join(LOG_DIR, "train-test-log.json" if test else "train-log.json")

    if not os.path.exists(log_file):
        with open(log_file, 'w') as file:
            json.dump([], file)

    with open(log_file, 'r') as file:
        log = json.load(file)
    
    log.append(log_entry)
    
    with open(log_file, 'w') as file:
        json.dump(log, file)
    
    print(f"Train log updated for {tag} from {dates[0]} to {dates[1]}")

def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=False):
    """
    update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=False)
    
    Update predict log file
    """
    log_entry = {
        "timestamp": str(datetime.now()), 
        "country": country, 
        "y_pred": y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred, 
        "y_proba": y_proba.tolist() if isinstance(y_proba, np.ndarray) else y_proba,
        "target_date": target_date, 
        "runtime": runtime, 
        "model_version": model_version
    }

    log_file = os.path.join(LOG_DIR, "predict-test-log.json" if test else "predict-log.json")

    if not os.path.exists(log_file):
        with open(log_file, 'w') as file:
            json.dump([], file)

    try:
        with open(log_file, 'r') as file:
            log = json.load(file)
    except json.JSONDecodeError:
        log = []
    
    log.append(log_entry)
    
    with open(log_file, 'w') as file:
        json.dump(log, file)
    
    print(f"Prediction log updated for {country} on {target_date}")