# Iris classification API

A small FastAPI example that trains a Random Forest on the Iris dataset and
serves predictions over HTTP.

## Setup

From the repository root:

```bash
python -m pip install -e ".[notebooks,api]"
```

## Train and serve

```bash
cd notebooks/iris_api
python train_model.py
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs.

The training script writes `model.joblib` and `scaler.joblib` locally (gitignored).
