from src.entity.config_entity import ModelTrainerConfig
from src.utils.model import * 
from src.utils.flower_dataset import *
from tqdm.auto import tqdm 
import os 

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = UNet()

    def train(self):
        if not os.path.exists(self.config.root_dir):
            os.makedirs(self.config.root_dir, exist_ok=True)
        train_dataloader = get_dataloader(self.config.train_data_dir)
        val_dataloader = get_dataloader(self.config.val_data_dir)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scheduler = LinearScheduler(0.0001, 0.02, 1000, device)
        if device == 'cuda':
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
            self.model.cuda()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss().to(device)
        epochs = 1
        best_val_loss = 1e9

        print("***START TRAINING***")
        for epoch in tqdm(range(epochs)):
            self.model.train()
            running_train_loss, running_val_loss = 0., 0.
            for idx, (original_L_images, original_AB_images) in enumerate(tqdm(train_dataloader)):
                original_L_images = original_L_images.to(device)
                original_AB_images = original_AB_images.to(device)
                
                batch_size = original_L_images.shape[0]
                noise = torch.randn_like(original_AB_images, device=device)
                time_step = torch.randint(0, 1000, (batch_size,)).long().to(device)
                noise_image = scheduler.add_noise(original_AB_images, noise, time_step)
                input_image = torch.cat((original_L_images, noise_image), dim=1)
                optimizer.zero_grad()
                pred_noise = self.model(input_image, time_step)
                train_loss = loss_fn(pred_noise, noise)
                running_train_loss += train_loss.item()
                train_loss.backward()
                optimizer.step()
                break 

                
            self.model.eval()
            with torch.no_grad():
                for idx, (original_L_images, original_AB_images) in enumerate(tqdm(val_dataloader)):
                    original_L_images = original_L_images.to(device)
                    original_AB_images = original_AB_images.to(device)
                    batch_size = original_L_images.shape[0]
                    noise = torch.randn_like(original_AB_images)
                    time_step = torch.randint(0, 1000, (batch_size,)).long().to(device)
                    noise_image = scheduler.add_noise(original_AB_images, noise, time_step)
                    input_image = torch.cat((original_L_images, noise_image), dim=1)
                    
                    pred_noise = self.model(input_image, time_step)
                    val_loss = loss_fn(pred_noise, noise)
                    running_val_loss += val_loss.item()
                    break 
                    
            running_train_loss /= len(train_dataloader)
            running_val_loss /= len(val_dataloader)
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {running_train_loss:.4f} - Val Loss: {running_val_loss:.4f}')
            if running_val_loss < best_val_loss:
                best_val_loss = running_val_loss
                torch.save(self.model.state_dict(), self.config.best_model_file)
                print('Saved Best Model')
                cnt = 0
            else:
                cnt += 1
                print(f'Early Stopping: {cnt}/10')
                if cnt == 10:
                    print('Stop training......')
                    break
        print("***END TRAINING***")
        
        
        