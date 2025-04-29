# 🌸 Flowers-Colorization using DDPM

This project applies **Denoising Diffusion Probabilistic Models (DDPM)** to the task of **automatic colorization** of flower images with MLOps Pipeline.  
By learning a diffusion-based generative process, the model can transform grayscale (black-and-white) flower images into realistic and vivid colorized versions.

---

## ✨ Project Overview

- **Goal**:  
  To develop a model that takes a grayscale image of a flower and generates a realistic colorized image.

- **Approach**:  
  We employ a **DDPM** architecture that learns to reverse a gradual noising process, starting from a noise distribution to generate a colorful, detailed image conditioned on the input grayscale image.

- **Dataset**:  
  We use a dataset of flower images on kaggle (https://www.kaggle.com/datasets/imsparsh/flowers-dataset), preprocessing it.

---

## 🛠️ Project Structure

```bash
Flowers-Colorization/
│
├── .github/                # Github action
├── config/              # Config for MLOps pipeline
├── src/              # All the souce code
├── requirements.txt     # Python dependencies
└── README.md            # Project description (you are here!)
```

---

## 🧠 Model Details

- **Backbone**:  
  A **U-Net** model is used for the noise prediction network within the DDPM framework.

- **Conditioning**:  
  The model conditions on grayscale images during both training and sampling.

- **Training Objective**:  
  Minimize the mean squared error (MSE) between the predicted noise and the true noise at different timesteps.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

You can modify the data path in `config.py` accordingly.

### 2. Run Full Pipeline

```bash
python main.py
```

Training progress (losses, sample generations) will be logged for monitoring.

### 3. Generate Colorized Flowers

After training:

```bash
python -m src.utils.inference
```

This will produce colorized versions of the grayscale flower images.

---

## 🧩 Main Features

- **Flexible**: Easily adapt to different flower datasets or resolution sizes.
- **Configurable**: Hyperparameters like noise schedule, number of diffusion steps, learning rate, batch size are adjustable via `config.py`.
- **Visualization**: Intermediate colorizations during sampling are saved for analysis.

---

## 📈 Results

Sample colorized outputs:

| Grayscale Input | Colorized Output |
|:---------------:|:----------------:|
| ![0_gray](https://github.com/user-attachments/assets/84f596f2-2973-44a6-a8b8-be21fab2b8e2) | ![2488902131_3417698611_n](https://github.com/user-attachments/assets/df1bf19a-446f-4be1-a80f-0b040e1f7f71) |
| ![1_gray](https://github.com/user-attachments/assets/0a6241c4-afed-41cc-ab61-409f637cc0c8) | ![2498632196_e47a472d5a](https://github.com/user-attachments/assets/25edbf1c-b19e-4fa2-bb26-bdf5b1517910) |

---

## ⚙️ Requirements

- Python >= 3.8
- PyTorch >= 1.10
- torchvision
- numpy
- tqdm
- matplotlib
- (optional) wandb or tensorboard for experiment tracking

See `requirements.txt` for full details.

---

## 📚 References

- **DDPM Paper**: [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- **U-Net Paper**: [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)

---
