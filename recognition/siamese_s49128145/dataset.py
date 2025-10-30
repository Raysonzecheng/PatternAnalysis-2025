import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import pandas as pd

class ISICDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.label_map = { 'melanoma': 1, 'normal': 0 }

    def __getitem__(self, idx):
        img1_path = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        label1 = self.label_map[self.data.iloc[idx, 1]]

        # 随机选一个同类或不同类图片
        should_match = random.randint(0, 1)
        while True:
            idx2 = random.randint(0, len(self.data) - 1)
            label2 = self.label_map[self.data.iloc[idx2, 1]]
            if (should_match and label1 == label2) or (not should_match and label1 != label2):
                break

        img2_path = os.path.join(self.image_folder, self.data.iloc[idx2, 0])
        img1, img2 = Image.open(img1_path).convert('RGB'), Image.open(img2_path).convert('RGB')
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, torch.tensor(float(label1 != label2), dtype=torch.float32)

    def __len__(self):
        return len(self.data)
