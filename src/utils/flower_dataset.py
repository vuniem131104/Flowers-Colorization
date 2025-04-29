from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import os
import cv2 
import numpy as np
import torch 

class FlowerDataset(Dataset):
    def __init__(self, root_dir, size=(64, 64)):
        self.root_dir = root_dir
        self.size = size
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.root_dir, self.images[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = cv2.resize(image, self.size)
        L, A, B = cv2.split(image) 
        AB = np.stack([A, B], axis=0)
        L = torch.tensor(L, dtype=torch.float32) / 255.0
        L = 2 * L - 1
        L = L.unsqueeze(dim=0)
        AB = torch.tensor(AB, dtype=torch.float32) / 255.0
        AB = 2 * AB - 1
        return L, AB 
    
def get_dataloader(dataset_dir, batch_size=2, shuffle=True):
    dataset = FlowerDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
