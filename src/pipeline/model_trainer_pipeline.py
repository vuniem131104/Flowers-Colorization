from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer

class ModelTrainerPipeline:
    def __init__(self):
        pass
    
    def initiate_model_trainer_pipeline(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()