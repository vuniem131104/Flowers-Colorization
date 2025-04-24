from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation


class DataTransformationPipeline:
    def __init__(self):
        pass 
    
    def initiate_data_transformation_pipeline(self):
        try:
            configmanager = ConfigurationManager()
            data_transformation_config = configmanager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.train_test_split()
        except Exception as e:
            raise e
