import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResNet50_SelfReg(nn.Module):
    def __init__(self, n_classes=2, pretrained=True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        input_feat_size = hidden_size

        self.cdpl = nn.Sequential(
                nn.Linear(input_feat_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, input_feat_size),
                nn.BatchNorm1d(input_feat_size)
            )
        
        self.require_all_grads()

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        outputs = self.fc(self.dropout(self.relu(features)))
        return outputs, features