import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        base_model = models.resnet18(pretrained=True)
        base_model.fc = nn.Identity()
        self.feature_extractor = base_model
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distances = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(distances, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2))
        return loss
