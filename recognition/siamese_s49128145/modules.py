import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
import random
import torch.nn.functional as F

###############################################################################
### Triplet Network
class TripletNetwork(nn.Module):
    """
    Triplet network based on ResNet50 backbone.
    """
    def __init__(self, emb_dim=128):
        super(TripletNetwork, self).__init__()
        resnet = resnet50(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, emb_dim),
        )

    def forward_once(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return F.normalize(out, p=2, dim=1)

    def forward(self, anchor, positive, negative):
        emb_a = self.forward_once(anchor)
        emb_p = self.forward_once(positive)
        emb_n = self.forward_once(negative)
        return emb_a, emb_p, emb_n

###############################################################################
# Step 2: Binary Classifier
class BinaryClassifier(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

###############################################################################
### Config Settings
def get_config() -> dict:
    """
    Configuration for binary classification training.
    """
    config = {
        'data_subset': 1000,  # or None to use all data
        'metadata_path': './data/train-metadata.csv',
        'image_dir': './data/train-image/image/',
        'batch_size': 16,
        'embedding_dims': 128,   # kept for consistency, though not needed
        'learning_rate': 1e-4,
        'epochs': 20,
    }
    return config


###############################################################################
def set_seed(seed: int = 42) -> None:
    

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

