import torch
import torch.nn as nn
import torchvision
from models.classification.MLP import MLP

class ResNet50_DANN(nn.Module):
    def __init__(self, pretrained=True, hidden_size=2048, dropout=0.5, n_classes=2, n_domains=2):
        super().__init__()

        self.update_count = 0
        
        # Algorithms
        self.featurizer = torchvision.models.resnet50(pretrained=pretrained)
        self.featurizer.fc = nn.Linear(2048, hidden_size)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes))

        self.discriminator = MLP(hidden_size, n_domains, dropout)

        self.class_embeddings = nn.Embedding(n_classes, hidden_size)

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        return self.classifier(self.featurizer(x))