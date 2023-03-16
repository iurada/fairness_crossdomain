import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50 import ResNet50

class Pretrain:
    def __init__(self):
        pass

    def start(self):
        pass

class Experiment:
    def __init__(self, args):
        self.args = args

        # Model setup
        self.model = ResNet50()
        self.model.to(DEVICE)
        self.model.train()

        # Optimizer setup
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)

        self.iteration = 0
        self.best_metric = 0.0

    def save(self, path, iteration, best_metric):
        self.iteration = iteration
        self.best_metric = best_metric

        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iteration': self.iteration,
            'best_metric': self.best_metric
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iteration = checkpoint['iteration']
        self.best_metric = checkpoint['best_metric']
        return self.iteration, self.best_metric

    def train_iteration(self, data):
        x, y, g = data
        x, y = x.to(DEVICE), y.to(DEVICE)

        pred, _ = self.model(x)

        loss = F.cross_entropy(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'train_loss': loss.item()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        predicted = []
        target = []
        group = []

        for x, y, g in loader:
            x, y, g = x.to(DEVICE), y.to(DEVICE), g.to(DEVICE)

            pred, _ = self.model(x)

            predicted.append(pred)
            target.append(y)
            group.append(g)
        
        predicted = torch.cat(predicted)
        target = torch.cat(target)
        group = torch.cat(group)

        self.model.train()
        
        return predicted, target, group