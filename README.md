# CI/CD Pipeline: House Price Prediction

A small end-to-end MLOps example. A scikit-learn linear regression model is served by a FastAPI backend, used by a Streamlit front end, tested on every push with GitHub Actions and deployed to Render.

The model predicts house price from **area (sqft)** and **number of bedrooms**. The data is a tiny demo set, so the point of the project is the pipeline, not the prediction.

## How it fits together

```
data/data.csv --> backend/training.py --> backend/models/model.pkl
                                              |
                       FastAPI backend  <-----+      (backend/main.py)
                             ^
                             |  POST /predict
                       Streamlit front end           (frontend/app.py)
```

## Run it locally

Use Python 3.10 or newer.

**1. Train the model** (optional, a trained `model.pkl` is already included):

```bash
pip install pandas scikit-learn
python backend/training.py
```

**2. Start the API:**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

The API runs at http://127.0.0.1:8000 (interactive docs at `/docs`).

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Health check, returns `{"status": "API running"}` |
| `POST /predict` | Body `{"area": 1200, "bedrooms": 2}`, returns the predicted price |

**3. Start the front end** (in another terminal):

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

By default the app calls the deployed API. To use your local one, change `API_URL` in `frontend/app.py` to `http://127.0.0.1:8000/predict`.

## Tests

```bash
cd backend
pip install pytest httpx
python -m pytest tests/
```

## CI/CD

Two GitHub Actions workflows live in `.github/workflows/`:

| Workflow | Runs when | What it does |
| --- | --- | --- |
| `backend-ci.yaml` | Push to `main` | Installs dependencies, checks the app imports, runs the tests, then triggers a Render deploy |
| `frontend-ci.yaml` | Push to `main` that changes `frontend/**` | Installs dependencies, checks Streamlit imports, then triggers a Render deploy |

Deploys use Render deploy hooks stored as repository secrets: `RENDER_DEPLOY_HOOK` (backend) and `RENDER_FRONTEND_DEPLOY_HOOK` (front end). If a secret is missing, the deploy step is skipped.

## Project structure

```
backend/
  main.py            FastAPI app
  training.py        Trains and saves the model
  models/model.pkl   Trained model
  tests/             API tests
data/data.csv        Demo training data
frontend/app.py      Streamlit app
.github/workflows/   CI/CD pipelines
```

## Author

Dr. Enosh A Paulson. [LinkedIn](https://linkedin.com/in/dr-enosh) · [GitHub](https://github.com/DrEnosh)
