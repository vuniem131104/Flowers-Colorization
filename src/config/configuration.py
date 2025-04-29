from dataclasses import dataclass 
from src.constants import *
from src.utils.common import read_yaml, create_directories  
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.schema=read_yaml(schema_filepath)
        self.params=read_yaml(params_filepath)
        create_directories([self.config.artifact_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        root_dir = self.config.data_ingestion.root_dir
        local_file = self.config.data_ingestion.local_file
        dataset_name = self.config.data_ingestion.dataset_name
        local_dataset_dir = self.config.data_ingestion.local_dataset_dir

        data_ingestion_config = DataIngestionConfig(
            root_dir=root_dir,
            local_file=local_file,
            dataset_name=dataset_name,
            local_dataset_dir=local_dataset_dir
        )
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        data_transformation_config=DataTransformationConfig(
            root_dir=self.config.data_transformation.root_dir,
            data_dir=self.config.data_transformation.data_dir,
            train_data_dir=self.config.data_transformation.train_data_dir,
            test_data_dir=self.config.data_transformation.test_data_dir,
            val_data_dir=self.config.data_transformation.val_data_dir
        )
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        model_trainer_config=ModelTrainerConfig(
            root_dir=self.config.model_trainer.root_dir,
            train_data_dir=self.config.model_trainer.train_data_dir,
            val_data_dir=self.config.model_trainer.val_data_dir,
            best_model_file=self.config.model_trainer.best_model_file,
            batch_size=self.config.model_trainer.batch_size,
            image_size=self.config.model_trainer.image_size
        )
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        model_evaluation_config=ModelEvaluationConfig(
            root_dir=self.config.model_evaluation.root_dir,
            best_model_file=self.config.model_evaluation.best_model_file,
            test_data_dir=self.config.model_evaluation.test_data_dir,
            result_file=self.config.model_evaluation.result_file
        )
        return model_evaluation_config