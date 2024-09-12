import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd

class MedicalImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.data.iloc[idx, 0]}"
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.mask_dir:
            mask_path = f"{self.mask_dir}/{self.data.iloc[idx, 1]}"
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.0  # Normalize mask
            mask = mask.astype(float)
        else:
            mask = None

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            if mask is not None:
                mask = augmented['mask']
        
        if mask is not None:
            return image, mask
        return image

transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
              std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

dataset = MedicalImageDataset(
    csv_file='data/labels.csv',
    img_dir='data/images',
    mask_dir='data/masks',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
