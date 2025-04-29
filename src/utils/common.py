import os
import yaml
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError
from torch import nn 
import torch 


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            print(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
        


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    print(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    print(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    print(f"binary file loaded from: {path}")
    return data

class LinearScheduler:
    def __init__(self, beta_start, beta_end, steps):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.steps = steps 
        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0] 

        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t].to(original.device).reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t].to(original.device).reshape(batch_size, 1, 1, 1)

        return sqrt_alphas_cumprod * original + sqrt_one_minus_alphas_cumprod * noise

    def sample_prev_sample(self, xt, t, noise_pred):
        x0 = (xt - self.sqrt_one_minus_alphas_cumprod.to(xt.device)[t] * noise_pred) / self.sqrt_alphas_cumprod.to(xt.device)[t] 
        x0 = torch.clamp(x0, -1, 1)
        mean = (xt - self.betas.to(xt.device)[t] * noise_pred) / self.sqrt_one_minus_alphas_cumprod.to(xt.device)[t]
        if t == 0:
            return mean, x0 
        else:
            variance = (1 - self.alphas_cumprod.to(xt.device)[t - 1]) / (1.0 - self.alphas_cumprod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0 
        
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:] 
        new_state_dict[name] = v
    return new_state_dict