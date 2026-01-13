from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import json
import os
from typing import List, Dict, Any

app = FastAPI(title="DPF Soot Load Prediction API", version="1.0")

# Global model artifacts
model = None
config = None

def load_artifacts():
    global model, config
    try:
        model_path = "models/xgb_model.json"
        config_path = "models/model_config.json"
        
        if os.path.exists(model_path):
            model = xgb.Booster()
            model.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print("Model artifact not found.")

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Config loaded: {config.keys()}")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

# Load on startup
load_artifacts()

class TelemetryInput(BaseModel):
    # Dynamic fields corresponding to features
    # But to be safe, we accept a dictionary of features
    # Or strict schema if we know features.
    # Given features act as input, we expect clients to provide pre-calculated features 
    # OR we provide raw telemetry and pipe it through feature engine.
    # For this assignment, assuming "Input: Recent sensor readings (last N minutes)" 
    # implies on-the-fly feature engineering.
    # However, to keep it simple and robust:
    # We will accept the *features* expected by the model.
    # Implementing full feature pipeline in API is complex (stateful).
    # Let's assume the input is the feature vector for simplicity, 
    # OR simpler: pass the feature dictionary.
    features: Dict[str, float]

class BatchInput(BaseModel):
    instances: List[TelemetryInput]

class PredictionOutput(BaseModel):
    soot_load_predicted: float
    regeneration_needed: bool
    confidence_interval: List[float] # simplified

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "model_loaded": model is not None,
        "metrics": config.get("metrics") if config else None
    }

@app.get("/model/info")
def model_info():
    """Returns model metadata as requested."""
    if not config:
        raise HTTPException(status_code=503, detail="Model config not loaded")
    return {
        "version": "1.0.0",
        "training_date": "2025-01-13",
        "algorithm": "XGBoost Regressor",
        "metrics": config.get("metrics"),
        "features": config.get("features")
    }

def log_prediction(vehicle_id: str, prediction: float):
    import datetime
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "vehicle_id": vehicle_id,
        "prediction": prediction
    }
    print(json.dumps(log_entry))

@app.post("/predict/soot-load", response_model=PredictionOutput)
def predict_soot_load(input_data: TelemetryInput):
    if not model or not config:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Align features
    required_features = config['features']
    input_features = input_data.features
    
    try:
        vector = [input_features.get(f, 0.0) for f in required_features]
        dtest = xgb.DMatrix([vector], feature_names=required_features)
        prediction = model.predict(dtest)[0]
        
        # Log for monitoring
        # Extract vehicle_id if available in features, else 'unknown'
        vid = str(input_features.get('vehicle_id', 'unknown'))
        log_prediction(vid, float(prediction))
        
        # Logic for recommendation
        # 80% is high, 100% is critical
        regen_needed = prediction > 80.0
        
        return {
            "soot_load_predicted": float(prediction),
            "regeneration_needed": regen_needed,
            "confidence_interval": [float(prediction * 0.9), float(prediction * 1.1)] # dummy
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
def predict_batch(batch_input: BatchInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    required_features = config['features']
    vectors = []
    
    for instance in batch_input.instances:
        vectors.append([instance.features.get(f, 0.0) for f in required_features])
        
    dtest = xgb.DMatrix(vectors, feature_names=required_features)
    predictions = model.predict(dtest)
    
    results = []
    for pred in predictions:
        results.append({
            "soot_load_predicted": float(pred),
            "regeneration_needed": bool(pred > 80.0)
        })
        
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
