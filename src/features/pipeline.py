import pandas as pd
import numpy as np
from typing import List, Optional

class FeaturePipeline:
    def __init__(self):
        pass
        
    def load_data(self, data_dir: str = 'data') -> pd.DataFrame:
        """Loads and joins telemetry and maintenance data."""
        telemetry = pd.read_csv(f'{data_dir}/telemetry.csv')
        maintenance = pd.read_csv(f'{data_dir}/maintenance.csv')
        
        # Convert timestamps
        telemetry['timestamp'] = pd.to_datetime(telemetry['timestamp'])
        maintenance['timestamp'] = pd.to_datetime(maintenance['timestamp'])
        
        return telemetry, maintenance

    def transform(self, telemetry: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering logic."""
        df = telemetry.copy()
        
        # Sort is critical for rolling windows and merge_asof
        # merge_asof requires sorting by the 'on' key (timestamp)
        df.sort_values('timestamp', inplace=True)
        
        # 1. Rolling Features
        # Group by vehicle to avoid bleeding data across vehicles
        vehicle_group = df.groupby('vehicle_id')
        
        # Exhaust Temp Rolling Stats (Short and Medium term)
        # 5 min = 1 step, 15 min = 3 steps, 60 min = 12 steps
        windows = [3, 12] 
        for w in windows:
            # ffill() ensures we use the last valid reading if current is missing, 
            # simulating a "hold" or simply handling gaps before rolling
            df[f'exhaust_temp_pre_mean_{w*5}m'] = vehicle_group['exhaust_temp_pre'].transform(
                lambda x: x.ffill().rolling(window=w, min_periods=1).mean()
            )
            df[f'diff_pressure_mean_{w*5}m'] = vehicle_group['diff_pressure'].transform(
                lambda x: x.ffill().rolling(window=w, min_periods=1).mean()
            )
            
        # 2. Cumulative Features (Since last Regen)
        # We need to identify regen events and create a 'reset' signal
        # Merging maintenance events onto telemetry
        maint_subset = maintenance[['vehicle_id', 'timestamp', 'type']].copy()
        maint_subset.rename(columns={'timestamp': 'regen_time'}, inplace=True)
        
        # Use asof merge to find the *latest* regen event before current timestamp
        df = pd.merge_asof(
            df, 
            maint_subset.sort_values('regen_time'), 
            left_on='timestamp', 
            right_on='regen_time', 
            by='vehicle_id', 
            direction='backward'
        )
        
        # Calculate time since regen
        # If no regen found (NaN), assume start of data is the reference point (or some default large value)
        # Here we fill NaN regen_time with vehicle's first timestamp
        df['regen_time'] = df['regen_time'].fillna(df.groupby('vehicle_id')['timestamp'].transform('min'))
        
        df['time_since_last_regen_hours'] = (df['timestamp'] - df['regen_time']).dt.total_seconds() / 3600.0
        
        # Cumulative Distance and Load since regen
        # We need cumulative sums, resetting at regen_time.
        # This is tricky in pandas vectorized. 
        # Easier approach: Global cumsum - Global cumsum at regen_time
        
        # Calculate global cumulative stats per vehicle
        # Approx distance per step: speed (km/h) * (5/60) h
        df['step_dist_km'] = df['speed'] * (5/60.0)
        df['global_cum_dist'] = df.groupby('vehicle_id')['step_dist_km'].cumsum()
        
        # Get the global_cum_dist AT the time of regen
        # Since we merged regen_time, we can merge the global_cum_dist value at that time too?
        # A simpler way: 'Regen Group' ID. 
        # Create a signal where regen happens, cumsum that signal to get a group ID.
        
        # Re-approach for robustness:
        # Create a 'regen_event' column in telemetry
        # We can map maintenance timestamps to the nearest telemetry timestamp
        events = maintenance.set_index(['vehicle_id', 'timestamp'])
        # Depending on precision, we might need to round telemetry timestamps to match, or use `merge_asof` again purely for event identification
        
        # Let's stick to the 'time since' logic which is already correct.
        # For 'distance since', we can do:
        # DistSince = GlobalCumDist(Current) - GlobalCumDist(RegenTime)
        
        # We need to lookup GlobalCumDist at RegenTime.
        # Let's create a lookup table of RegenTime -> GlobalCumDist
        # But RegenTime comes from maintenance table, which doesn't have the cumulative stats yet.
        # We can interpolate or nearest-match.
        
        # Alternative:
        # data['regen_count'] = data.groupby('vehicle').apply(...)
        # This is slow for large data.
        
        # Fast vectorised approach:
        # 1. merge_asof (already done) gives us 'regen_time' for every row.
        # 2. We need 'dist_at_last_regen'. 
        #    We can build a map: (vehicle_id, regen_time) -> global_cum_dist_at_that_time
        #    But global_cum_dist is defined on telemetry grid.
        #    So we find the telemetry row equal to OR just before regen_time, get its cum_dist.
        
        # Let's get the distinct regen times from our enriched DF
        regen_lookup = df[['vehicle_id', 'regen_time', 'global_cum_dist']].drop_duplicates(subset=['vehicle_id', 'regen_time'], keep='last')
        
        # This lookup might be wrong if 'regen_time' in the DF is repeating for many rows (it is).
        # We want the cum_dist at the moment that regen_time *started*.
        # So we want the first row where `timestamp >= regen_time`? 
        # Or if we merged 'backward', we have the regen_time.
        # Actually, `merge_asof` propagates the value forward.
        # So for a whole block of rows sharing the same `regen_time`, they share the same reference point.
        # The reference cumulative distance IS the cumulative distance of the row corresponding to `regen_time`.
        # BUT `regen_time` might not exactly match a telemetry timestamp.
        
        # Let's simplify: 
        # Just compute sum of distance grouped by (vehicle, regen_time).
        df['dist_since_last_regen'] = df.groupby(['vehicle_id', 'regen_time'])['step_dist_km'].cumsum()

        # Same for Engine Load (integrated load)
        df['step_load'] = df['engine_load'] * (5/60.0) # Load-hours? Or just sum of load. 
        # Taking integral of load over time.
        df['load_sum_since_last_regen'] = df.groupby(['vehicle_id', 'regen_time'])['step_load'].cumsum()

        # 3. Ratio Features
        df['pressure_per_rpm'] = df['diff_pressure'] / (df['rpm'] + 1.0) # Avoid div/0
        
        # Clean up intermediate cols
        drop_cols = ['step_dist_km', 'step_load', 'global_cum_dist']
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        # Handle NaN created by lags/rolling
        df.fillna(0, inplace=True)
        
        return df

if __name__ == "__main__":
    import os
    if os.path.exists('data/telemetry.csv'):
        pipeline = FeaturePipeline()
        tel, maint = pipeline.load_data()
        processed_df = pipeline.transform(tel, maint)
        print("Features created:", processed_df.shape)
        print(processed_df.columns)
        processed_df.to_csv('data/processed_features.csv', index=False)
    else:
        print("Data not found. Run generator.py first.")
