import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

class DataGenerator:
    def __init__(self, num_vehicles: int = 10, start_date: str = '2025-01-01', duration_days: int = 30):
        self.num_vehicles = num_vehicles
        self.start_date = pd.to_datetime(start_date)
        self.duration_days = duration_days
        self.end_date = self.start_date + timedelta(days=duration_days)
        self.vehicle_ids = [f'V{str(i).zfill(3)}' for i in range(1, num_vehicles + 1)]
        
        # Physics simulation constants
        self.base_soot_rate = 0.05  # % per hour
        self.regen_threshold_passive = 350.0  # Temp C where passive regen starts
        self.regen_rate_passive = 1.5  # % per hour reduction
        self.active_regen_threshold_soot = 80.0  # % soot load to trigger active regen
        self.active_regen_temp = 600.0  # Temp C during active regen
        
    def generate_all(self, output_dir: str = 'data'):
        """Generates all datasets and saves them to CSV."""
        print(f"Generating data for {self.num_vehicles} vehicles over {self.duration_days} days...")
        
        telemetry_dfs = []
        maintenance_records = []
        trips = []
        
        for vid in self.vehicle_ids:
            df_tel, records, vehicle_trips = self._simulate_vehicle(vid)
            telemetry_dfs.append(df_tel)
            maintenance_records.extend(records)
            trips.extend(vehicle_trips)
            
        # Combine and Save
        telemetry_all = pd.concat(telemetry_dfs, ignore_index=True)
        maintenance_all = pd.DataFrame(maintenance_records)
        trips_all = pd.DataFrame(trips)
        
        # Sort by time (check if not empty to avoid KeyError)
        if not telemetry_all.empty:
            telemetry_all.sort_values(['vehicle_id', 'timestamp'], inplace=True)
        
        if not maintenance_all.empty:
            maintenance_all.sort_values(['vehicle_id', 'timestamp'], inplace=True)
        else:
            # Ensure columns exist even if empty, for downstream pipeline
            maintenance_all = pd.DataFrame(columns=['vehicle_id', 'timestamp', 'type'])
            
        if not trips_all.empty:
            trips_all.sort_values(['vehicle_id', 'trip_start'], inplace=True)
        
        # Save to disk
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        telemetry_all.to_csv(f'{output_dir}/telemetry.csv', index=False)
        maintenance_all.to_csv(f'{output_dir}/maintenance.csv', index=False)
        trips_all.to_csv(f'{output_dir}/trips.csv', index=False)
        
        print(f"Data generation complete. Saved to {output_dir}/")
        return telemetry_all, maintenance_all, trips_all

    def _simulate_vehicle(self, vehicle_id: str) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
        """Simulates a single vehicle's operation."""
        # Time steps: 5 minute intervals for telemetry
        freq = '5T' 
        timestamps = pd.date_range(start=self.start_date, end=self.end_date, freq=freq)
        n_steps = len(timestamps)
        
        # calibrated random walks for base signals
        np.random.seed(hash(vehicle_id) % 2**32) # Deterministic per vehicle
        
        # Speed & Load Profiles (correlated)
        # Using a Markov chain-like approach or simple smoothed random walk
        speed_base = np.random.normal(loc=40, scale=20, size=n_steps) # km/h
        speed_base = np.clip(pd.Series(speed_base).rolling(window=12, min_periods=1).mean().values, 0, 100)
        
        # Idle periods (speed = 0)
        is_idle = np.random.random(n_steps) < 0.2
        speed_base[is_idle] = 0.0
        
        # Engine Load correlates with speed but has noise
        engine_load = speed_base * 0.8 + np.random.normal(10, 5, n_steps)
        engine_load[is_idle] = np.random.normal(5, 2, np.sum(is_idle)) # Idle load
        engine_load = np.clip(engine_load, 0, 100)

        # Ambient Temp
        ambient_temp = np.random.normal(20, 5, n_steps) # 20C mean
        
        # --- Physics Simulation Loop ---
        soot_load = []
        exhaust_temp_pre = []
        exhaust_temp_post = []
        diff_pressure = []
        rpm = []
        
        current_soot = np.random.uniform(0, 30) # Start with some soot
        maintenance_events = []
        
        # State tracking for Active Regen
        in_active_regen = False
        active_regen_duration = 0
        
        for i in range(n_steps):
            spd = speed_base[i]
            load = engine_load[i]
            amb = ambient_temp[i]
            
            # RPM model
            if spd == 0:
                curr_rpm = 600 + np.random.normal(0, 10)
            else:
                curr_rpm = 600 + (spd * 25) + np.random.normal(0, 50)
            
            # Exhaust Temp Model
            # Higher load -> Higher Temp
            target_temp = 150 + (load * 4) # Max ~550C at 100% load
            curr_exhaust_temp = target_temp + np.random.normal(0, 10)
            
            # Soot Accumulation
            # High load + Low Temp -> High Soot features (incomplete combustion)
            # But mostly soot accumulates over time/usage
            soot_production = (load / 100.0) * 0.1 # Base production proportional to load
            
            # Passive Regen
            passive_burn = 0
            if curr_exhaust_temp > self.regen_threshold_passive:
                # Exponentially better burn at higher temps
                passive_burn = ((curr_exhaust_temp - self.regen_threshold_passive) / 100.0) ** 2 * 0.2
            
            # Active Regen Logic
            if current_soot > self.active_regen_threshold_soot and not in_active_regen and spd > 10:
                # Trigger active regen
                in_active_regen = True
                active_regen_duration = 6 # 30 mins (6 * 5min)
                maintenance_events.append({
                    'vehicle_id': vehicle_id,
                    'timestamp': timestamps[i],
                    'type': 'active_regeneration',
                    'trigger_soot_level': current_soot
                })
            
            if in_active_regen:
                curr_exhaust_temp = self.active_regen_temp + np.random.normal(0, 15)
                # Fast burn
                soot_reduction = 5.0 # % per 5 min
                current_soot -= soot_reduction
                active_regen_duration -= 1
                if active_regen_duration <= 0 or current_soot < 5:
                    in_active_regen = False
            else:
                # Normal operation
                current_soot += soot_production - passive_burn
            
            current_soot = np.clip(current_soot, 0, 120) # Max 120% (clogged)
            
            # Differential Pressure Model (The Proxy)
            # Pressure = f(Flow, Soot)
            # Flow is proportional to RPM/Load
            flow_factor = (curr_rpm / 2000.0) * (1 + load/200.0)
            base_pressure = flow_factor * (1 + current_soot/50.0) # Pressure rises with soot
            curr_pressure = base_pressure + np.random.normal(0, 0.05)
            
            # Post-DPF Temp (lagged and slightly cooler unless regen)
            if in_active_regen:
                curr_post_temp = curr_exhaust_temp + 50 # Exothermic reaction in DPF
            else:
                curr_post_temp = curr_exhaust_temp - 20 # Heat loss
            
            # Simulate Sensor Failures (Missing Data)
            # Randomly drop 1% of readings for robustness testing
            if np.random.random() < 0.01: curr_pressure = np.nan
            if np.random.random() < 0.01: curr_rpm = np.nan
            if np.random.random() < 0.01: curr_exhaust_temp = np.nan
                
            soot_load.append(current_soot)
            exhaust_temp_pre.append(curr_exhaust_temp)
            exhaust_temp_post.append(curr_post_temp)
            diff_pressure.append(max(0, curr_pressure) if not np.isnan(curr_pressure) else np.nan)
            rpm.append(curr_rpm)
            
        # Create DataFrame
        df = pd.DataFrame({
            'vehicle_id': vehicle_id,
            'timestamp': timestamps,
            'engine_load': engine_load,
            'speed': speed_base,
            'rpm': rpm,
            'exhaust_temp_pre': exhaust_temp_pre,
            'exhaust_temp_post': exhaust_temp_post,
            'diff_pressure': diff_pressure,
            'ambient_temp': ambient_temp,
            'soot_load_ground_truth': soot_load # Target variable
        })
        
        # Add Trip Data (Simplified)
        # Identify non-zero speed blocks
        df['is_moving'] = df['speed'] > 0.5
        df['trip_start'] = (df['is_moving'] & ~df['is_moving'].shift(1).fillna(False))
        trip_starts = df[df['trip_start']].index
        
        trips = []
        for start_idx in trip_starts:
            # Find end
            # Look ahead for stop
            future_stop = df.loc[start_idx:].index[~df.loc[start_idx:, 'is_moving']]
            if len(future_stop) > 0:
                end_idx = future_stop[0]
            else:
                end_idx = df.index[-1]
                
            trip_slice = df.loc[start_idx:end_idx]
            if len(trip_slice) < 5: continue # Ignore micro moves
            
            trips.append({
                'vehicle_id': vehicle_id,
                'trip_start': df.loc[start_idx, 'timestamp'],
                'trip_end': df.loc[end_idx, 'timestamp'],
                'distance_km': (trip_slice['speed'].mean() * (len(trip_slice) * 5 / 60)), # avg speed * hours
                'avg_speed': trip_slice['speed'].mean(),
                'max_speed': trip_slice['speed'].max(),
                'idle_time_pct': (trip_slice['speed'] < 5).mean()
            })
            
        return df, maintenance_events, trips

if __name__ == "__main__":
    gen = DataGenerator(num_vehicles=5, duration_days=30)
    gen.generate_all()
