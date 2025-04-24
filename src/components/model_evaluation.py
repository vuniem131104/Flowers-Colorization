from src.entity.config_entity import ModelEvaluationConfig
import pandas as pd
import os
from joblib import load
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import json
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        if not os.path.exists(self.config.root_dir):
            os.makedirs(self.config.root_dir, exist_ok=True)
        test_df = pd.read_csv(self.config.test_data_file)
        X_test = test_df.drop(columns=['quality'])
        y_test = test_df['quality']
        model = load(self.config.model_path)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        result = {
            "mae": mae,
            "r2": r2,
            "mse": mse
        }
        with open(self.config.result_file, 'w') as f:
            json.dump(result, f)
        
        print(f"Model Evaluation Results: \nMAE: {mae}\nR2: {r2}\nMSE: {mse}")
        