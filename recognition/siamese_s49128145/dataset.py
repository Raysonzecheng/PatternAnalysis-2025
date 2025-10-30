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

def get_isic2020_data(metadata_path: str, image_dir: str, data_subset: int | None=None) -> tuple[list]:
    """
    Returns the images and associated labels for the isic 2020 data set.
    Returns: images, labels
    """
    metadata = pd.read_csv(metadata_path)

    metadata['image_file'] = metadata['isic_id'] + '.jpg'

    image_to_label = dict(zip(metadata['image_file'], metadata['target']))
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img in image_to_label]

    if data_subset:
        pos_paths = [img for img in image_paths if image_to_label[os.path.basename(img)] == 1][:data_subset // 2]
        neg_paths = [img for img in image_paths if image_to_label[os.path.basename(img)] == 0][:data_subset // 2]
        image_paths = pos_paths + neg_paths

    labels = [image_to_label[os.path.basename(path)] for path in image_paths]
    return np.array(image_paths), np.array(labels)

def train_val_test_split_with_augmentation(images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray]:
    
    # 1️⃣ 基本划分
    train_images, other_images, train_labels, other_labels = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )
    test_images, val_images, test_labels, val_labels = train_test_split(
        other_images, other_labels, test_size=0.5, stratify=other_labels, random_state=42
    )

    # 2️⃣ 分类
    class_0_images = train_images[train_labels == 0]
    class_1_images = train_images[train_labels == 1]
    class_0_labels = train_labels[train_labels == 0]
    class_1_labels = train_labels[train_labels == 1]

    # 3️⃣ 计算需要扩增的数量
    num_class_0 = len(class_0_images)
    num_class_1 = len(class_1_images)
    diff = num_class_0 - num_class_1

    if diff > 0:
        print(f"⏫ Augmenting {diff} melanoma images to balance dataset...")
        new_images = []
        new_labels = []

        # 从 class_1_images 随机挑选并增强
        for _ in range(diff):
            idx = random.randint(0, num_class_1 - 1)
            img_path = class_1_images[idx]
            new_path = augment_image(img_path)
            new_images.append(new_path)
            new_labels.append(1)

        # 合并增强后的图片
        class_1_images = np.concatenate([class_1_images, np.array(new_images)], axis=0)
        class_1_labels = np.concatenate([class_1_labels, np.array(new_labels)], axis=0)

    # 4️⃣ 合并最终训练集
    train_images = np.concatenate([class_0_images, class_1_images], axis=0)
    train_labels = np.concatenate([class_0_labels, class_1_labels], axis=0)

    print(f"✅ Balanced training set: {len(train_images)} samples "
          f"({sum(train_labels==0)} normal, {sum(train_labels==1)} melanoma)")

    return train_images, val_images, test_images, train_labels, val_labels, test_labels

def get_isic2020_data_loaders(images, labels, train_bs=32, test_val_bs=320, aug_factor=1):
    train_images, val_images, test_images, train_labels, val_labels, test_labels = train_val_test_split_with_augmentation(images, labels)

    train_ds = ClassificationDataset(
        images=train_images,
        labels=train_labels,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=10 * aug_factor, fill=(255, 255, 255)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1 * aug_factor, contrast=0.1 * aug_factor, saturation=0.1 * aug_factor, hue=0.05 * aug_factor),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    )

    val_ds = ClassificationDataset(
        images=val_images,
        labels=val_labels,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    )

    test_ds = ClassificationDataset(
        images=test_images,
        labels=test_labels,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    )

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=test_val_bs, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=test_val_bs, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader



