"""
This module prepares data for training a Triplet Network on the ISIC 2020 dataset.
It includes:
- TripletDataset: PyTorch dataset class that returns (anchor, positive, negative) triplets
- get_isic2020_data: Loads image paths and labels from metadata
- train_val_test_split_with_augmentation: Splits data and oversamples minority class
- get_isic2020_data_loaders: Builds DataLoaders with augmentation
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

###############################################################################
# TripletDataset: Dataset class for Triplet Network
###############################################################################
class TripletDataset(Dataset):
    """
    PyTorch dataset for Triplet Network training.
    Returns: (anchor, positive, negative, anchor_label)
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.transform = transform

        # Group indices by class for sampling
        self.class_to_indices = {
            0: np.where(self.labels == 0)[0],
            1: np.where(self.labels == 1)[0]
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Anchor image and label
        anchor_path = self.images[idx]
        anchor_label = self.labels[idx]

        # Sample positive (same class)
        pos_idx = np.random.choice(self.class_to_indices[anchor_label])
        pos_path = self.images[pos_idx]

        # Sample negative (different class)
        neg_class = 1 - anchor_label
        neg_idx = np.random.choice(self.class_to_indices[neg_class])
        neg_path = self.images[neg_idx]

        # Load and convert images to RGB
        anchor = cv2.cvtColor(cv2.imread(str(anchor_path)), cv2.COLOR_BGR2RGB)
        positive = cv2.cvtColor(cv2.imread(str(pos_path)), cv2.COLOR_BGR2RGB)
        negative = cv2.cvtColor(cv2.imread(str(neg_path)), cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        else:
            to_tensor = transforms.ToTensor()
            anchor = to_tensor(anchor)
            positive = to_tensor(positive)
            negative = to_tensor(negative)

        return anchor, positive, negative, anchor_label


###############################################################################
# get_isic2020_data: Load image paths and labels 
###############################################################################
def get_isic2020_data(metadata_path: str, image_dir: str, data_subset: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ISIC 2020 image paths and labels.
    Args:
        metadata_path: Path to CSV metadata file
        image_dir: Directory containing image files
        data_subset: Optional number of samples to load (balanced)
    Returns:
        Tuple of image paths and labels as numpy arrays
    """
    metadata = pd.read_csv(metadata_path)
    metadata['image_file'] = metadata['isic_id'].astype(str) + '.jpg'
    image_to_label = dict(zip(metadata['image_file'], metadata['target']))

    # Sort filenames for deterministic order
    filenames = sorted(os.listdir(image_dir))
    image_paths = [os.path.join(image_dir, f) for f in filenames if f in image_to_label]

    # Optional balanced subset sampling
    if data_subset:
        pos_paths = [p for p in image_paths if image_to_label[os.path.basename(p)] == 1][:data_subset // 2]
        neg_paths = [p for p in image_paths if image_to_label[os.path.basename(p)] == 0][:data_subset // 2]
        image_paths = pos_paths + neg_paths

    labels = [image_to_label[os.path.basename(p)] for p in image_paths]
    return np.array(image_paths), np.array(labels)


###############################################################################
# train_val_test_split_with_augmentation: Split and oversample training data
###############################################################################
def train_val_test_split_with_augmentation(images: np.ndarray, labels: np.ndarray, aug_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets and oversample minority class in training.
    Args:
        images: Array of image paths
        labels: Array of binary labels
        aug_dir: Optional directory for saving augmented images (unused)
    Returns:
        Train, val, test image paths and labels
    """
    
    # Initial split: 80% train, 20% other
    train_images, other_images, train_labels, other_labels = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )
    # Split remaining 20% into 10% val and 10% test
    test_images, val_images, test_labels, val_labels = train_test_split(
        other_images, other_labels, test_size=0.5, stratify=other_labels, random_state=42
    )

    # Separate classes in training set
    class_0_images = train_images[train_labels == 0]
    class_0_labels = train_labels[train_labels == 0]
    class_1_images = train_images[train_labels == 1]
    class_1_labels = train_labels[train_labels == 1]

    num_class_0 = len(class_0_images)
    num_class_1 = len(class_1_images)
    print(f"Before oversampling: class_0 = {num_class_0}, class_1 = {num_class_1}")

    # Oversample minority class
    if num_class_1 < num_class_0:
        oversample_indices = np.random.choice(
            np.arange(num_class_1), size=num_class_0 - num_class_1, replace=True
        )
        oversampled_class_1_images = class_1_images[oversample_indices]
        oversampled_class_1_labels = class_1_labels[oversample_indices]

        class_1_images = np.concatenate([class_1_images, oversampled_class_1_images], axis=0)
        class_1_labels = np.concatenate([class_1_labels, oversampled_class_1_labels], axis=0)

    # Merge and shuffle training set
    train_images = np.concatenate([class_0_images, class_1_images], axis=0)
    train_labels = np.concatenate([class_0_labels, class_1_labels], axis=0)

    shuffle_idx = np.random.permutation(len(train_images))
    train_images = train_images[shuffle_idx]
    train_labels = train_labels[shuffle_idx]

    print(f"After oversampling: class_0 = {np.sum(train_labels == 0)}, class_1 = {np.sum(train_labels == 1)}")

    return train_images, val_images, test_images, train_labels, val_labels, test_labels

###############################################################################
# get_isic2020_data_loaders: Build PyTorch DataLoaders
###############################################################################
def get_isic2020_data_loaders(images, labels, train_bs=32, test_val_bs=320, aug_factor=1, aug_dir: Optional[str] = None):
    """
    Build PyTorch DataLoaders for training, validation, and testing.
    Args:
        images: Image paths
        labels: Binary labels
        train_bs: Batch size for training
        test_val_bs: Batch size for validation and testing
        aug_factor: Augmentation intensity multiplier
        aug_dir: Optional directory for saving augmented images (unused)
    Returns:
        train_loader, val_loader, test_loader
    """

    train_images, val_images, test_images, train_labels, val_labels, test_labels = train_val_test_split_with_augmentation(
        images, labels, aug_dir=aug_dir
    )
    
    # Training transformations
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

    # Evaluation transformations
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Build datasets
    train_ds = TripletDataset(train_images, train_labels, transform=train_transform)
    val_ds = TripletDataset(val_images, val_labels, transform=eval_transform)
    test_ds = TripletDataset(test_images, test_labels, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=test_val_bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=test_val_bs, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

