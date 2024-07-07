import warnings
warnings.filterwarnings("ignore", message="resource_tracker: .*")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import joblib
from typing import Optional
from model.model import model_train, model_predict, model_load

sys.path.append("model")

app = FastAPI()

class TrainRequest(BaseModel):
    data_dir: str
    test: Optional[bool] = False

class PredictRequest(BaseModel):
    country: str
    year: str
    month: str
    day: str
    test: Optional[bool] = False

@app.post("/train")
def train_model(request: TrainRequest):
    try:
        model_train(request.data_dir, test=request.test)
        return {"message": "Training completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_model(request: PredictRequest):
    try:
        result = model_predict(request.country, request.year, request.month, request.day, test=request.test)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)