from src.entity.config_entity import DataTransformationConfig
from src.config.configuration import ConfigurationManager
import pandas as pd
import os
import random
import shutil
        
class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_split(self, test_size: float = 0.2):
        if not os.path.exists(self.config.root_dir):
            os.makedirs(self.config.root_dir, exist_ok=True)
        if not os.path.exists(self.config.train_data_dir):
            os.makedirs(self.config.train_data_dir, exist_ok=True)
        if not os.path.exists(self.config.test_data_dir):
            os.makedirs(self.config.test_data_dir, exist_ok=True)

        list_images = os.listdir(self.config.data_dir)
        random.shuffle(list_images)
        train_size = int(len(list_images) * (1 - test_size))
        train_images = list_images[:train_size]
        test_images = list_images[train_size:]

        for image in train_images:
            src = os.path.join(self.config.data_dir, image)
            dst = os.path.join(self.config.train_data_dir, image)
            shutil.move(src, dst)
        for image in test_images:
            src = os.path.join(self.config.data_dir, image)
            dst = os.path.join(self.config.test_data_dir, image)
            shutil.move(src, dst)


if __name__ == "__main__":
    config = ConfigurationManager()
    data_transformation = DataTransformation(config=config.get_data_transformation_config())
    data_transformation.train_test_split()
        
                
        
