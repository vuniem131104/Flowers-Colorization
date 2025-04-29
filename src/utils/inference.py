from src.utils.model import UNet
import torch
import cv2 
from src.utils.model import LinearScheduler
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch 
from src.utils.common import remove_module_prefix
from tqdm.auto import tqdm

model = UNet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load("artifacts/model_trainer/model.pth", map_location=device)
if device == 'cuda':
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()
else:
    state_dict = remove_module_prefix(state_dict)

model.load_state_dict(state_dict)
scheduler = LinearScheduler(0.0001, 0.02, 1000, device)

model.eval()
gray_image = cv2.imread('/home/vuiem/ColorDiffusion/artifacts/data_transformation/test/2539552964_921cf645ba_n.jpg')
gray_image = cv2.resize(gray_image, (64, 64))
lab_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2Lab)
l_image, _, _ = cv2.split(lab_image)
l_image = torch.tensor(l_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0 
l_image = 2 * l_image - 1
l_image = l_image.to(device)
ab_image = torch.randn(1, 2, 64, 64).to(device)
image = torch.cat([l_image, ab_image], dim=1)
with torch.inference_mode():
    for i in tqdm(reversed(range(1000))):
        time_step = torch.as_tensor(i).unsqueeze(0).to(device)
        noise_prediction = model(image , time_step)
        ab_image = scheduler.sample_prev_timestep(ab_image, noise_prediction, time_step)[0]
        image = torch.cat([l_image, ab_image], dim=1)

lab_image = (image[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8) 
rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
plt.imshow(rgb_image)
plt.show()