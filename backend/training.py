from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "data.csv"
MODEL_PATH = ROOT / "backend" / "models" / "model.pkl"

df = pd.read_csv(DATA_PATH)

X = df[["area", "bedrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {MODEL_PATH}")
