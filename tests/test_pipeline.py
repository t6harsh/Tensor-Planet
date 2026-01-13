import pytest
import pandas as pd
import numpy as np
import os
from src.data.generator import DataGenerator
from src.features.pipeline import FeaturePipeline

@pytest.fixture
def sample_data():
    gen = DataGenerator(num_vehicles=2, duration_days=2)
    return gen.generate_all(output_dir='data_test')

def test_data_generation(sample_data):
    tel, maint, trips = sample_data
    assert not tel.empty
    assert 'soot_load_ground_truth' in tel.columns
    assert 'vehicle_id' in tel.columns
    # Check soot range
    assert tel['soot_load_ground_truth'].min() >= 0
    assert tel['soot_load_ground_truth'].max() <= 120 # Capped

def test_pipeline_transform(sample_data):
    tel, maint, trips = sample_data
    pipeline = FeaturePipeline()
    
    # Simulate loading (convert dtypes) - CRITICAL for merge_asof
    tel['timestamp'] = pd.to_datetime(tel['timestamp'])
    maint['timestamp'] = pd.to_datetime(maint['timestamp'])
    
    processed = pipeline.transform(tel, maint)
    
    # Check features
    expected_cols = ['exhaust_temp_pre_mean_15m', 'dist_since_last_regen', 'pressure_per_rpm']
    for col in expected_cols:
        assert col in processed.columns
        
    # Check that sorting was maintained/enforced
    assert processed['timestamp'].is_monotonic_increasing

def test_feature_pipeline_logic():
    # Synthetic small case
    tel = pd.DataFrame({
        'vehicle_id': ['V1', 'V1', 'V1'],
        'timestamp': pd.to_datetime(['2025-01-01 10:00', '2025-01-01 10:05', '2025-01-01 10:10']),
        'exhaust_temp_pre': [100, 200, 300],
        'diff_pressure': [10, 20, 30],
        'rpm': [1000, 1000, 1000],
        'speed': [50, 50, 50],
        'engine_load': [50, 50, 50]
    })
    
    maint = pd.DataFrame({
        'vehicle_id': ['V1'],
        'timestamp': pd.to_datetime(['2025-01-01 10:00']),
        'type': ['regen']
    })
    
    pipeline = FeaturePipeline()
    res = pipeline.transform(tel, maint)
    
    # Check cumulative time
    # 10:05 - 10:00 = 5 mins = 0.0833 hours
    assert abs(res.iloc[1]['time_since_last_regen_hours'] - 0.08333) < 0.001
    
    # Check rolling mean (window=3 is 15m, here we have 3 points)
    # last point rolling mean of temp: (100+200+300)/3 = 200
    assert abs(res.iloc[2]['exhaust_temp_pre_mean_15m'] - 200.0) < 0.1

def test_edge_case_new_dpf():
    """Test behavior for a brand new DPF (no history)."""
    tel = pd.DataFrame({
        'vehicle_id': ['V_NEW'],
        'timestamp': pd.to_datetime(['2025-01-01 12:00']),
        'exhaust_temp_pre': [200],
        'diff_pressure': [0.5],
        'rpm': [800],
        'speed': [0],
        'engine_load': [10]
    })
    # Empty maintenance records
    maint = pd.DataFrame(columns=['vehicle_id', 'timestamp', 'type'])
    
    # Cast timestamps for merge_asof compatibility
    tel['timestamp'] = pd.to_datetime(tel['timestamp'])
    maint['timestamp'] = pd.to_datetime(maint['timestamp'])
    
    pipeline = FeaturePipeline()
    res = pipeline.transform(tel, maint)
    
    # Needs to run without error
    assert not res.empty
    # Time since regen should be undefined or 0 or handled.
    # Logic: fillna with min timestamp. 
    # So time since regen = 0
    assert res.iloc[0]['time_since_last_regen_hours'] == 0.0

if __name__ == "__main__":
    pytest.main()
