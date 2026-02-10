from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Linear Regression API")

default_model_path = Path(__file__).resolve().parent / "models" / "model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(default_model_path)))
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


class InputData(BaseModel):
    area: float
    bedrooms: int


@app.post("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.area,data.bedrooms]])
    prediction = model.predict(X)[0]
    return {"predicted_price": float(prediction)}
