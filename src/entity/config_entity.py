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
    val_data_dir: str

@dataclass
class ModelTrainerConfig:
    root_dir: str  
    train_data_dir: str 
    val_data_dir: str 
    best_model_file: str 
    batch_size: int
    image_size: int 


@dataclass
class ModelEvaluationConfig:
    root_dir: str  
    best_model_file: str 
    test_data_dir: str 
    result_file: str
    