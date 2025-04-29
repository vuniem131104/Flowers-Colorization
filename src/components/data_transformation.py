from src.entity.config_entity import DataTransformationConfig
from src.config.configuration import ConfigurationManager
import pandas as pd
import os
import random
import shutil
        
class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_split(self, test_size: float = 0.1, val_size: float = 0.1):
        if not os.path.exists(self.config.root_dir):
            os.makedirs(self.config.root_dir, exist_ok=True)
        if not os.path.exists(self.config.train_data_dir):
            os.makedirs(self.config.train_data_dir, exist_ok=True)
        if not os.path.exists(self.config.test_data_dir):
            os.makedirs(self.config.test_data_dir, exist_ok=True)
        if not os.path.exists(self.config.val_data_dir):
            os.makedirs(self.config.val_data_dir, exist_ok=True)

        list_images = os.listdir(self.config.data_dir)
        random.shuffle(list_images)
        train_len = int(len(list_images) * (1 - test_size - val_size))
        test_len = int(len(list_images) * test_size)
        val_len = int(len(list_images) * val_size)
        train_images = list_images[:train_len]
        test_images = list_images[train_len:train_len + test_len]
        val_images = list_images[train_len + test_len:]

        for image in train_images:
            src = os.path.join(self.config.data_dir, image)
            dst = os.path.join(self.config.train_data_dir, image)
            shutil.copyfile(src, dst)
        for image in test_images:
            src = os.path.join(self.config.data_dir, image)
            dst = os.path.join(self.config.test_data_dir, image)
            shutil.copyfile(src, dst)
        for image in val_images:
            src = os.path.join(self.config.data_dir, image)
            dst = os.path.join(self.config.val_data_dir, image)
            shutil.copyfile(src, dst)

        
                
        
