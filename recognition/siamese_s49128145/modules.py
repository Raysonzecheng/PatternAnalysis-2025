"""
This module defines:
- TripletNetwork: A ResNet50-based embedding model for triplet learning
- BinaryClassifier: A classifier that operates on learned embeddings
- get_config: Centralized configuration dictionary for training
- set_seed: Utility for reproducible experiments
"""
import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
import random
import torch.nn.functional as F

###############################################################################
# TripletNetwork: Embedding model for triplet learning
###############################################################################
class TripletNetwork(nn.Module):
    """
    Triplet Network using ResNet50 as backbone.
    Outputs normalized embeddings for anchor, positive, and negative images.
    """
    def __init__(self, emb_dim=128):
        super(TripletNetwork, self).__init__()
        resnet = resnet50(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, emb_dim),
        )

    def forward_once(self, x):
        """
        Forward pass for a single image.
        Returns L2-normalized embedding.
        """
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return F.normalize(out, p=2, dim=1)

    def forward(self, anchor, positive, negative):
        """
        Forward pass for triplet input.
        Returns embeddings for anchor, positive, and negative.
        """
        emb_a = self.forward_once(anchor)
        emb_p = self.forward_once(positive)
        emb_n = self.forward_once(negative)
        return emb_a, emb_p, emb_n
    
    def encode(self, x):
        """
        Encode a single image or batch into embedding space.
        """
        return self.forward_once(x)

###############################################################################
# BinaryClassifier: Classifier on top of embeddings
###############################################################################
class BinaryClassifier(nn.Module):
    """
    Fully connected classifier for binary prediction from embeddings.
    """
    def __init__(self, emb_dim=128):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(emb_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)   # Output logits for binary classification
        )

    def forward(self, x):
        return self.fc_layers(x)

###############################################################################
# get_config: Centralized configuration dictionary
###############################################################################
def get_config() -> dict:
    """
    Returns configuration dictionary for training and evaluation.
    """
    config = {
        'data_subset': 5000,  # Set to None to use full dataset
        'metadata_path': './data/train-metadata.csv',
        'image_dir': './data/train-image/image/',
        'batch_size': 16,
        'embedding_dims': 128,   
        'learning_rate': 1e-4,
        'epochs': 20,
    }
    return config


###############################################################################
# set_seed: Ensure reproducibility across libraries
###############################################################################
def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across NumPy, Python, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

