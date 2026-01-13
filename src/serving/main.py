from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional

app = FastAPI(title="DPF Soot Load Prediction API", version="2.0")

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

class TelemetryReading(BaseModel):
    timestamp: str # ISO format
    exhaust_temp_pre: float
    diff_pressure: float
    rpm: float
    engine_load: float
    speed: float

class VehicleState(BaseModel):
    # Cumulative metrics tracked by the vehicle ECU or fleet system
    time_since_last_regen_hours: float
    dist_since_last_regen_km: float

class PredictionRequest(BaseModel):
    vehicle_id: str
    # Window of recent data (e.g. last 60 mins), ordered by time
    history_window: List[TelemetryReading]
    current_state: VehicleState

class PredictionOutput(BaseModel):
    vehicle_id: str
    soot_load_predicted: float
    regeneration_needed: bool
    confidence_interval: List[float]
    warnings: List[str] = []

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "model_loaded": model is not None,
        "metrics": config.get("metrics") if config else None,
        "api_version": "2.0 (Raw Telemetry Support)"
    }

@app.get("/model/info")
def model_info():
    """Returns model metadata."""
    if not config:
        raise HTTPException(status_code=503, detail="Model config not loaded")
    return {
        "version": "1.0.0",
        "training_date": "2025-01-13",
        "algorithm": "XGBoost Regressor",
        "metrics": config.get("metrics"),
        "features": config.get("features")
    }

def calculate_features(window: List[TelemetryReading], state: VehicleState) -> Dict[str, float]:
    """Computes model features from raw history window."""
    if not window:
        raise ValueError("Empty history window")
        
    # Convert to DataFrame
    data = [r.dict() for r in window]
    df = pd.DataFrame(data)
    
    # 1. Rolling Features
    # We take the rolling mean of the END of the window.
    # If window is shorter than 60 mins, we take what we have.
    
    # Pre-process: handle explicit None/NaN if pydantic allowed them (it enforces float, but worth noting)
    # The pipeline logic used ffill() then rolling()
    
    # Stats: 15m (3 steps of 5m) and 60m (12 steps)
    # Ensure sorted
    # We assume the last item in 'window' is the "current" timestamp for prediction
    
    # Calculate means
    # If we have 12 rows, mean is simple.
    features = {}
    
    # Helper for rolling mean of last N items
    def get_rolling_mean(col, n):
        series = df[col]
        if len(series) == 0: return 0.0
        # take last min(n, len) items
        window_slice = series.tail(n) 
        return float(window_slice.mean())

    # 15 min ~ 3 samples (assuming 5 min frequency)
    # 60 min ~ 12 samples
    features['exhaust_temp_pre_mean_15m'] = get_rolling_mean('exhaust_temp_pre', 3)
    features['exhaust_temp_pre_mean_60m'] = get_rolling_mean('exhaust_temp_pre', 12)
    
    features['diff_pressure_mean_15m'] = get_rolling_mean('diff_pressure', 3)
    features['diff_pressure_mean_60m'] = get_rolling_mean('diff_pressure', 12)
    
    # 2. Cumulative Features (passed in state)
    features['dist_since_last_regen'] = state.dist_since_last_regen_km
    features['time_since_last_regen_hours'] = state.time_since_last_regen_hours
    
    # 3. Ratio Features (on current/last reading)
    last_row = df.iloc[-1]
    features['engine_load'] = float(last_row['engine_load'])
    features['rpm'] = float(last_row['rpm'])
    
    rpm_safe = features['rpm'] + 1.0 # 0 division protection
    # Use the smoothed pressure for stability, or raw? Pipeline used raw diff_pressure column
    # Pipeline: df['pressure_per_rpm'] = df['diff_pressure'] / (df['rpm'] + 1.0)
    features['pressure_per_rpm'] = float(last_row['diff_pressure']) / rpm_safe
    
    return features

@app.post("/predict/soot-load", response_model=PredictionOutput)
def predict_soot_load(request: PredictionRequest):
    if not model or not config:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    required_features = config['features']
    warnings = []
    
    try:
        # Compute features
        feats = calculate_features(request.history_window, request.current_state)
        
        # Prepare vector
        # Handle missing features (defaults)
        vector = []
        for f in required_features:
            if f not in feats:
                # Some features might be missing if pipeline changed
                # Safe default 0
                vector.append(0.0)
            else:
                vector.append(feats[f])
                
        dtest = xgb.DMatrix([vector], feature_names=required_features)
        prediction = model.predict(dtest)[0]
        pred_val = float(prediction)
        
        # Logic for recommendation
        regen_needed = pred_val > 80.0
        
        return {
            "vehicle_id": request.vehicle_id,
            "soot_load_predicted": pred_val,
            "regeneration_needed": regen_needed,
            "confidence_interval": [pred_val * 0.9, pred_val * 1.1],
            "warnings": warnings
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
