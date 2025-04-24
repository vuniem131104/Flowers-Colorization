from dataclasses import dataclass 
from zipfile import ZipFile  
from src.constants import *
from src.utils.common import read_yaml, create_directories  
import urllib.request as request
import os 
from src.entity.config_entity import DataIngestionConfig
import kagglehub
import os 
import shutil

class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        unzip_dir = self.config.unzip_dir
        os.makedirs(unzip_dir, exist_ok=True)
        if not os.path.exists(self.config.local_file):
            path = kagglehub.dataset_download(self.config.dataset_name)
            print(f"File downloaded! with following info: \n{path}")
        else:
            print(f"File already exists!!!")

    def move_files(self):
        local_file = self.config.local_file
        unzip_dir = self.config.unzip_dir
        for item in os.listdir(local_file):
            s = os.path.join(local_file, item)
            d = os.path.join(unzip_dir, item)
            shutil.move(s, d)
        print(f"Files moved!!!")

    def remove_files(self):
        unzip_dir = self.config.unzip_dir
        for item in os.listdir(unzip_dir):
            if os.path.isfile(os.path.join(unzip_dir, item)):
                os.remove(os.path.join(unzip_dir, item))
        print(f"Files removed!!!")

    # def join_folder_to_dataset(self):
    #     unzip_dir = self.config.unzip_dir
    #     for item in os.listdir(unzip_dir):
    #         if 
    #     print(f"Folders joined!!!")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with ZipFile(file=self.config.local_data_file, mode="r") as zip_ref:
            zip_ref.extractall(path=unzip_path)
        print(f"Zip file extracted!!!")
        
# if __name__ == "__main__":
#     configmanager = ConfigurationManager()
#     data_ingestion_config = configmanager.get_data_ingestion_config()
#     data_ingestion = DataIngestion(config=data_ingestion_config)
#     data_ingestion.download_file()
#     data_ingestion.extract_zip_file()