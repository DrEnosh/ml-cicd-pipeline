from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Linear Regression API")

default_model_path = Path(__file__).resolve().parent / "models" / "model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(default_model_path)))
model = None


class InputData(BaseModel):
    area: float
    bedrooms: int


@app.get("/")
def health_check():
    return {"status": "API running"}


@app.post("/predict")
def predict(data: InputData):
    global model
    if model is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model from {MODEL_PATH}: {e}")
    X = np.array([[data.area, data.bedrooms]])
    prediction = model.predict(X)[0]
    return {"predicted_price": float(prediction)}
