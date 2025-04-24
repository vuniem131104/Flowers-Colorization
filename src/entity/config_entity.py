from pathlib import Path 
from dataclasses import dataclass 

@dataclass
class DataIngestionConfig:
    root_dir: str 
    local_file: Path 
    dataset_name: str 
    local_dataset_dir: Path 

@dataclass
class DataTransformationConfig:
    root_dir: str  
    data_dir: str 
    train_data_dir: str 
    test_data_dir: str 

@dataclass
class ModelTrainerConfig:
    root_dir: str  
    train_data_file: str 
    model_name: str 
    alpha: float 
    l1_ratio: float 
    model_path: str 

@dataclass
class ModelEvaluationConfig:
    root_dir: str  
    test_data_file: str 
    model_path: str 
    result_file: str 
    