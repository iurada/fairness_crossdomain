import torchvision
import torch.nn as nn

class ResNet50_L2D(nn.Module):
    def __init__(self, n_classes=1, pretrained=True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Linear(2048, hidden_size)

        # Classifier
        self.fc = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.p_logvar = nn.Sequential(nn.Linear(2048, hidden_size), nn.ReLU())
        self.p_mu = nn.Sequential(nn.Linear(2048, hidden_size), nn.LeakyReLU())

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, train=True):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        # Collect stats
        logvar = self.p_logvar(x)
        mu = self.p_mu(x)

        if train: 
            # Reparametrize x
            factor = 0.2
            std = logvar.div(2).exp()
            eps = std.data.new(std.size()).normal_()
            x = mu + factor * std * eps
        else:
            x = mu
        
        embeddings = x
        # Classify
        outputs = self.fc(self.dropout(x))
        
        return outputs, logvar, mu, embeddings