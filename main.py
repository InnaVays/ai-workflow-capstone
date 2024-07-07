from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import json

# Import functions from model.py and cslib.py
from model import model_train, model_predict, model_load
from cslib import fetch_ts, engineer_features
from logger import update_train_log, update_predict_log

app = FastAPI()

class TrainRequest(BaseModel):
    data_dir: str
    test: bool = False

class PredictRequest(BaseModel):
    country: str
    year: str
    month: str
    day: str
    test: bool = False

class LogfileRequest(BaseModel):
    log_type: str
    tag: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.get("/favicon.ico")
def get_favicon():
    return HTTPException(status_code=404, detail="Favicon not found")

@app.post("/train")
def train_model(request: TrainRequest):
    try:
        model_train(request.data_dir, test=request.test)
        return {"message": "Model training completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_model(request: PredictRequest):
    try:
        result = model_predict(request.country, request.year, request.month, request.day, test=request.test)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/logfile")
def get_logfile(request: LogfileRequest):
    try:
        log_file = "train_log.json" if request.log_type == "train" else "predict_log.json"
        log_path = os.path.join("logs", log_file)
        
        if not os.path.exists(log_path):
            raise HTTPException(status_code=404, detail="Log file not found")
        
        with open(log_path, "r") as file:
            log_content = json.load(file)
        
        return {"log_content": log_content}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)