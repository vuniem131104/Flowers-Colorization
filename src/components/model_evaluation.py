from src.entity.config_entity import ModelTrainerConfig
from torch.utils.data import DataLoader
from src.utils.model import * 
from src.utils.flower_dataset import *
from src.utils.common import remove_module_prefix
from tqdm.auto import tqdm  
import torch 
from torch import nn 
import json 

class ModelEvaluation:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = UNet()
        
    def evaluate(self):
        if not os.path.exists(self.config.root_dir):
            os.makedirs(self.config.root_dir, exist_ok=True)
        loss_fn = nn.MSELoss()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scheduler = LinearScheduler(0.0001, 0.02, 1000, device)
        state_dict = torch.load(self.config.best_model_file, map_location=device)
        if device == 'cuda':
            model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
            model.cuda()
        else:
            state_dict = remove_module_prefix(state_dict)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        test_dataloader = get_dataloader(self.config.test_data_dir)
        test_loss = 0.0 
        with torch.no_grad():
            for idx, (original_L_images, original_AB_images) in enumerate(tqdm(test_dataloader)):
                original_L_images = original_L_images.to(device)
                original_AB_images = original_AB_images.to(device)
                batch_size = original_L_images.shape[0]
                noise = torch.randn_like(original_AB_images)
                time_step = torch.randint(0, 1000, (batch_size,)).long().to(device)
                noise_image = scheduler.add_noise(original_AB_images, noise, time_step)
                input_image = torch.cat((original_L_images, noise_image), dim=1)
                
                pred_noise = self.model(input_image, time_step)
                val_loss = loss_fn(pred_noise, noise)
                test_loss += val_loss.item()
                    
        test_loss /= len(test_dataloader)
        with open(self.config.result_file, 'w') as f:
            json.dump({'test_loss': test_loss}, f)
        
        
        