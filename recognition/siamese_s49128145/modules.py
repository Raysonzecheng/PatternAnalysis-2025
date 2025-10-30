import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np

class BinaryClassifier(nn.Module):
    """
    A binary classification model based on ResNet50.
    The ResNet backbone is used as a feature extractor, and a few custom
    fully connected layers are added for classification.
    """
    def __init__(self, emb_dim=128):
        super(BinaryClassifier, self).__init__()

        resnet = resnet50(weights="IMAGENET1K_V1")

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, 2)   
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out

def get_config() -> dict:
    config = {
        'data_subset': 1000,  
        'metadata_path': './data/train-metadata.csv',
        'image_dir': './data/train-image/image/',
        'batch_size': 16,
        'learning_rate': 1e-4,
        'epochs': 20,
    }
    return config


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
