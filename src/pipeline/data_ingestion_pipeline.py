from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion


class DataIngestionPipeline:
    def __init__(self):
        pass 
    
    def initiate_data_ingestion_pipeline(self):
        try:
            configmanager = ConfigurationManager()
            data_ingestion_config = configmanager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.move_files()
            data_ingestion.remove_files()
            data_ingestion.join_folder_to_dataset()
        except Exception as e:
            raise e
