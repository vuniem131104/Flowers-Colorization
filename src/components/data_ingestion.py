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
from src.config.configuration import ConfigurationManager

class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        root_dir = self.config.root_dir
        os.makedirs(root_dir, exist_ok=True)
        if not os.path.exists(self.config.local_file):
            path = kagglehub.dataset_download(self.config.dataset_name)
            print(f"File downloaded! with following info: \n{path}")
        else:
            print(f"File already exists!!!")

    def move_files(self):
        local_file = self.config.local_file
        root_dir = self.config.root_dir
        for item in os.listdir(local_file):
            s = os.path.join(local_file, item)
            d = os.path.join(root_dir, item)
            shutil.move(s, d)
        print(f"Files moved!!!")

    def remove_files(self):
        root_dir = self.config.root_dir
        for item in os.listdir(root_dir):
            if os.path.isfile(os.path.join(root_dir, item)):
                os.remove(os.path.join(root_dir, item))
        print(f"Files removed!!!")

    def join_folder_to_dataset(self):
        root_dir = self.config.root_dir
        local_dataset_dir = self.config.local_dataset_dir
        if not os.path.exists(local_dataset_dir):
            os.makedirs(local_dataset_dir, exist_ok=True)
        for item in os.listdir(root_dir):
            if item == 'test':
                for file in os.listdir(os.path.join(root_dir, item)):
                    shutil.move(os.path.join(root_dir, item, file), os.path.join(local_dataset_dir, file))
                shutil.rmtree(os.path.join(root_dir, item))
            elif item == 'train':
                for subfolder in os.listdir(os.path.join(root_dir, item)):
                    for file in os.listdir(os.path.join(root_dir, item, subfolder)):
                        shutil.move(os.path.join(root_dir, item, subfolder, file), os.path.join(local_dataset_dir, file))
                shutil.rmtree(os.path.join(root_dir, item))
        print(f"Folders joined!!!")
