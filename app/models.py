import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from flask import Flask, request, jsonify
from albumentations import Compose, RandomResizedCrop, HorizontalFlip, Normalize, RandomRotate90, ShiftScaleRotate, CoarseDropout
from albumentations.pytorch import ToTensorV2
from model import CustomCNN
from config import config


# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Define the model loading and prediction functions
def load_fold_models(num_folds, model_class, num_classes, model_paths):
    models = []
    for fold in range(1, num_folds + 1):
        model = model_class(num_classes=num_classes,config=config).to(device)
        checkpoint = torch.load(model_paths[fold - 1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set the model to evaluation mode
        models.append(model)
    return models

def predict_ensemble(models, image):
    with torch.no_grad():
        outputs = [model(image) for model in models]
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        _, predicted = torch.max(avg_output, 1)
    return predicted.item()


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image=np.array(image))['image']
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)


nclass=5
# Define the absolute base path to where the checkpoints are saved
base_model_path = "/Users/saahil/Desktop/Coding_Projects/DL/MicroscopicFungi"

# Define the paths to the saved models for each fold using the absolute path

model_paths = [os.path.join(base_model_path, f'checkpoint_fold{fold}_best.pth') for fold in range(1, config["num_folds"] + 1)]



# Load all the saved models
models = load_fold_models(config['num_folds'], CustomCNN, nclass, model_paths)
