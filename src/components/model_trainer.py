from src.entity.config_entity import ModelTrainerConfig
import pandas as pd 
from sklearn.linear_model import ElasticNet
from joblib import dump
import os
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_df = pd.read_csv(self.config.train_data_file)
        X_train = train_df.drop(columns=['quality'])
        y_train = train_df['quality']

        if self.config.model_name == "ElasticNet":
            model = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio)
        else:
            raise ValueError(f"Model {self.config.model_name} not found")
        model.fit(X_train, y_train)

        if not os.path.exists(self.config.root_dir):
            os.makedirs(self.config.root_dir, exist_ok=True)
        dump(model, self.config.model_path)
        print(f"Model saved at {self.config.model_path}")
        
        
        