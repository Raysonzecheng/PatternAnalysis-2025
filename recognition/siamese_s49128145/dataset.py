import os
import random
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
import os


###############################################################################
### Classes
class ClassificationDataset(torch.utils.data.Dataset):
    """
    Standard dataset generator for binary classification.
    Each item returns (image, label).
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx]) / 255.0
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

def augment_image(img_path):
    """对单张图像执行简单的数据增强并返回新图像路径"""
    img = Image.open(img_path).convert('RGB')

    # 随机翻转
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    if random.random() > 0.5:
        img = ImageOps.flip(img)

    # 随机旋转（±20°）
    angle = random.uniform(-20, 20)
    img = img.rotate(angle)

    # 随机亮度调整
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # 随机模糊
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.0)))

    # 保存到同目录下的新文件
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)
    name, ext = os.path.splitext(basename)
    new_path = os.path.join(dirname, f"{name}_aug{random.randint(1000,9999)}{ext}")
    img.save(new_path)

    return new_path