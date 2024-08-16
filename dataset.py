import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from albumentations import Compose, RandomResizedCrop, HorizontalFlip, RandomRotate90, ShiftScaleRotate, CoarseDropout, Normalize
from albumentations.pytorch import ToTensorV2

class FungiDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset='train'):
        self.root_dir = os.path.join(root_dir, subset)
        self.transform = transform
        self.classes = ['H1', 'H2', 'H3', 'H5', 'H6']
        self.image_paths, self.labels = self._load_dataset()

    def _load_dataset(self):
        image_paths = []  # Initialize image_paths as an empty list
        labels = []       # Initialize labels as an empty list
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.exists(cls_dir):
                raise FileNotFoundError(f"Directory {cls_dir} does not exist.")
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):
                    image_paths.append(img_path)
                    labels.append(label)
        return image_paths, labels  # Return both lists

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, label

def get_transforms(config):
    return Compose([
        RandomResizedCrop(config["height"], config["width"], scale=(0.8, 1.0)),
        HorizontalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30),
        CoarseDropout(max_holes=8, max_height=32, max_width=32),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
