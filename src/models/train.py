import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.xgboost
import pickle
import json
import os

class ModelTrainer:
    def __init__(self, data_path: str = 'data/processed_features.csv'):
        self.data_path = data_path
        self.target = 'soot_load_ground_truth'
        # Drop non-feature columns
        self.drop_cols = ['vehicle_id', 'timestamp', 'trip_start', 'trip_end', 'regen_time', 'soot_load_ground_truth', 'type']
        
    def train(self):
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based Split (Train on first 80% time, Test on last 20%)
        # Just simple split by date for the whole dataset might bleed vehicle info if not careful, 
        # but since all vehicles operate in parallel, splitting by time is correct for simulating "future" prediction.
        
        split_date = df['timestamp'].quantile(0.8)
        train_df = df[df['timestamp'] < split_date]
        test_df = df[df['timestamp'] >= split_date]
        
        print(f"Training set: {train_df.shape}, Test set: {test_df.shape}")
        
        X_train = train_df.drop(columns=self.drop_cols, errors='ignore')
        y_train = train_df[self.target]
        X_test = test_df.drop(columns=self.drop_cols, errors='ignore')
        y_test = test_df[self.target]
        
        # Save feature names for inference alignment
        feature_names = list(X_train.columns)
        
        # Train
        # Using MLFlow to track
        mlflow.set_experiment("DPF_Soot_Load_Prediction")
        
        with mlflow.start_run():
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'reg:squarederror',
                'n_jobs': -1
            }
            
            mlflow.log_params(params)
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            
            print(f"Test MAE: {mae:.4f}")
            print(f"Test RMSE: {rmse:.4f}")
            
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            
            # Feature Importance
            importance = model.feature_importances_
            feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)
            print("\nTop 10 Features:")
            print(feat_imp.head(10))
            
            # Save Model Artifacts
            os.makedirs('models', exist_ok=True)
            model.save_model("models/xgb_model.json")
            
            # Save metadata (features list)
            with open("models/model_config.json", "w") as f:
                json.dump({
                    "features": feature_names,
                    "metrics": {"mae": mae, "rmse": rmse}
                }, f)
                
            mlflow.xgboost.log_model(model, "model")
            print("Model saved to models/")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
