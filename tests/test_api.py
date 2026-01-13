import pytest
from unittest.mock import MagicMock, patch
from src.serving.main import predict_soot_load, calculate_features, TelemetryReading, VehicleState, PredictionRequest, app
import pandas as pd
import xgboost as xgb

# Mock Data
def get_valid_history():
    history = []
    base_time = pd.Timestamp("2026-01-01 10:00:00")
    for i in range(12): # 60 mins
        t = base_time + pd.Timedelta(minutes=5*i)
        history.append(TelemetryReading(
            timestamp=t.isoformat(),
            exhaust_temp_pre=300.0 + i*10, 
            diff_pressure=2.0 + i*0.1,
            rpm=1000.0,
            engine_load=50.0,
            speed=60.0
        ))
    return history

def get_valid_state():
    return VehicleState(
        time_since_last_regen_hours=10.0,
        dist_since_last_regen_km=500.0
    )

def test_calculate_features_logic():
    history = get_valid_history()
    state = get_valid_state()
    
    feats = calculate_features(history, state)
    
    # Check keys
    expected = [
        "exhaust_temp_pre_mean_15m", "exhaust_temp_pre_mean_60m",
        "diff_pressure_mean_15m", "diff_pressure_mean_60m",
        "dist_since_last_regen", "time_since_last_regen_hours",
        "pressure_per_rpm"
    ]
    for k in expected:
        assert k in feats
        
    # Verify values
    # Last temp is 410. Last 3 are 390, 400, 410. Mean = 400.
    assert feats['exhaust_temp_pre_mean_15m'] == 400.0
    # Last pressure is 3.1. Last 3 are 2.9, 3.0, 3.1. Mean = 3.0.
    assert abs(feats['diff_pressure_mean_15m'] - 3.0) < 0.001
    
    # Check cumulative
    assert feats['dist_since_last_regen'] == 500.0

@patch('src.serving.main.xgb')
@patch('src.serving.main.model')
@patch('src.serving.main.config')
def test_predict_endpoint(mock_config, mock_model, mock_xgb):
    # Setup Mocks
    mock_config.__getitem__.side_effect = lambda x: {
        'features': ['exhaust_temp_pre_mean_15m', 'dist_since_last_regen'] 
    } if x == 'features' else None
    
    # Mock DMatrix
    mock_dmatrix = MagicMock()
    mock_xgb.DMatrix.return_value = mock_dmatrix
    
    mock_booster = MagicMock()
    # model.predict returns np.array of floats
    mock_booster.predict.return_value = [85.5] 
    mock_model.predict = mock_booster.predict
    
    # Request
    req = PredictionRequest(
        vehicle_id="TEST_V1",
        history_window=get_valid_history(),
        current_state=get_valid_state()
    )
    
    response = predict_soot_load(req)
    
    # Verify response (returns dict when called directly)
    assert response["vehicle_id"] == "TEST_V1"
    assert response["soot_load_predicted"] == 85.5
    assert response["regeneration_needed"] == True # > 80.0
    
    # Verify model called correct
    assert mock_model.predict.called

def test_calculate_features_empty():
    with pytest.raises(ValueError):
        calculate_features([], get_valid_state())

