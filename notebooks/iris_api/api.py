from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
SCALER_PATH = APP_DIR / "scaler.joblib"

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise FileNotFoundError(
        "Model files not found. Run `python train_model.py` from notebooks/iris_api first."
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Create FastAPI app
app = FastAPI(title="ML Model API",
              description="API for making predictions with a machine learning model",
              version="1.0")

# Define input data model (based on your features)
class InputData(BaseModel):
    # Example for Iris dataset - modify for your own features
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

# Define prediction output model
class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    class_name: str  # Optional, if you have class names

# Create prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
async def predict(data: InputData):
    try:
        input_data = pd.DataFrame([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][prediction]
        
        # Get class name (if available)
        class_names = ["setosa", "versicolor", "virginica"]  # Replace with your class names
        class_name = class_names[prediction] if len(class_names) > prediction else "Unknown"
        
        # Return prediction
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "class_name": class_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {"message": "ML Model API is running. Use /predict endpoint for predictions."}

# Run the API
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)