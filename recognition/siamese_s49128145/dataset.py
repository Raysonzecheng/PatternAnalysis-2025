# dataset_fixed.py
import os
import random
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import torch
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


###############################################################################
### Dataset class for Triplet Network
class TripletDataset(Dataset):
    """
    Dataset for Triplet Network. Returns (anchor, positive, negative)
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.transform = transform

        # 按类别分组，方便采样
        self.class_to_indices = {
            0: np.where(self.labels == 0)[0],
            1: np.where(self.labels == 1)[0]
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # anchor
        anchor_path = self.images[idx]
        anchor_label = self.labels[idx]

        # 选正样本（同类）
        pos_idx = np.random.choice(self.class_to_indices[anchor_label])
        pos_path = self.images[pos_idx]

        # 选负样本（不同类）
        neg_class = 1 - anchor_label
        neg_idx = np.random.choice(self.class_to_indices[neg_class])
        neg_path = self.images[neg_idx]

        # 读取三张图片
        anchor = cv2.cvtColor(cv2.imread(str(anchor_path)), cv2.COLOR_BGR2RGB)
        positive = cv2.cvtColor(cv2.imread(str(pos_path)), cv2.COLOR_BGR2RGB)
        negative = cv2.cvtColor(cv2.imread(str(neg_path)), cv2.COLOR_BGR2RGB)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        else:
            to_tensor = transforms.ToTensor()
            anchor = to_tensor(anchor)
            positive = to_tensor(positive)
            negative = to_tensor(negative)

        return anchor, positive, negative


###############################################################################
### get_isic2020_data (unchanged but safer ordering)
def get_isic2020_data(metadata_path: str, image_dir: str, data_subset: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns image paths and labels arrays for ISIC 2020.
    """
    metadata = pd.read_csv(metadata_path)
    metadata['image_file'] = metadata['isic_id'].astype(str) + '.jpg'
    image_to_label = dict(zip(metadata['image_file'], metadata['target']))

    # iterate sorted filenames for deterministic order
    filenames = sorted(os.listdir(image_dir))
    image_paths = [os.path.join(image_dir, f) for f in filenames if f in image_to_label]

    if data_subset:
        # try to balance classes in the sampled subset
        pos_paths = [p for p in image_paths if image_to_label[os.path.basename(p)] == 1][:data_subset // 2]
        neg_paths = [p for p in image_paths if image_to_label[os.path.basename(p)] == 0][:data_subset // 2]
        image_paths = pos_paths + neg_paths

    labels = [image_to_label[os.path.basename(p)] for p in image_paths]
    return np.array(image_paths), np.array(labels)


###############################################################################
### train/val/test split with augmentation (safer)
def train_val_test_split_with_augmentation(images: np.ndarray, labels: np.ndarray, aug_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    0.8 train, 0.1 val, 0.1 test split.
    Oversample minority class in training set using augment_image_save (writes to aug_dir).
    Returns arrays for train/val/test images and labels.
    """
    # base split
    train_images, other_images, train_labels, other_labels = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=None
    )
    test_images, val_images, test_labels, val_labels = train_test_split(
        other_images, other_labels, test_size=0.5, stratify=other_labels, random_state=None
    )

    # separate classes in training set
    class_0_images = train_images[train_labels == 0]
    class_1_images = train_images[train_labels == 1]


    # compute how many to augment
    num_class_0 = len(class_0_images)
    num_class_1 = len(class_1_images)
    

    print(num_class_0,num_class_1)

   
    return train_images, val_images, test_images, train_labels, val_labels, test_labels


###############################################################################
### Data loader builder (fixed syntax / deterministic order)
def get_isic2020_data_loaders(images, labels, train_bs=32, test_val_bs=320, aug_factor=1, aug_dir: Optional[str] = None):
    train_images, val_images, test_images, train_labels, val_labels, test_labels = train_val_test_split_with_augmentation(
        images, labels, aug_dir=aug_dir
    )

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=10 * aug_factor, expand=False, fill=(255, 255, 255)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1 * aug_factor, contrast=0.1 * aug_factor, saturation=0.1 * aug_factor, hue=0.05 * aug_factor),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = TripletDataset(train_images, train_labels, transform=train_transform)
    val_ds = TripletDataset(val_images, val_labels, transform=eval_transform)
    test_ds = TripletDataset(test_images, test_labels, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=test_val_bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=test_val_bs, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

