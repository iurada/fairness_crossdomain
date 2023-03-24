import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

class SagResNet50(nn.Module):
    def __init__(self, n_classes=2, pretrained=True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet_c = torchvision.models.resnet50(pretrained=pretrained)
        self.resnet_s = torchvision.models.resnet50(pretrained=pretrained)

        # featurizer network
        self.network_f = nn.Sequential(
            self.resnet_c.conv1,
            self.resnet_c.bn1,
            self.resnet_c.relu,
            self.resnet_c.maxpool,
            self.resnet_c.layer1,
            self.resnet_c.layer2,
            self.resnet_c.layer3
        )
        # content network
        self.network_c = nn.Sequential(
            self.resnet_c.layer4,
            self.resnet_c.avgpool,
            nn.Flatten(),
            nn.Linear(2048, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )
        # style network
        self.network_s = nn.Sequential(
            self.resnet_s.layer4,
            self.resnet_s.avgpool,
            nn.Flatten(),
            nn.Linear(2048, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def randomize(self, x, what='style', eps=1e-5):
        # Implements the Style/Content Randomization Module (SR/CR)
        device = 'cuda' if x.is_cuda else 'cpu'
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)
        
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == 'style':
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()
        
        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def forward(self, x):
        return self.network_c(self.network_f(x))