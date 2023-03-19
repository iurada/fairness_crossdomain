import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

class ResNet50_Rotation(nn.Module):
    def __init__(self, n_classes=2, pretrained=True, hidden_size=2048, dropout=0.5, n_rot_classes=4, concat=True):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.concat = concat
        self.fc_rotation = nn.Linear(2 * hidden_size if concat else hidden_size, n_rot_classes)

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, x_rot=None):
        features = self.resnet(x)
        outputs = self.fc(self.dropout(self.relu(features)))
        
        if x_rot is not None:
            features_rot = self.resnet(x_rot)
        else:
            features_rot = features

        if self.concat:
            features_rot = torch.cat((features, features_rot), dim=-1)

        outputs_rot = self.fc_rotation(features_rot)

        return outputs, outputs_rot