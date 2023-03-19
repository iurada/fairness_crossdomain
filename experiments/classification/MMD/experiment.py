import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50 import ResNet50

def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1),
                      x1,
                      x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res.clamp_min_(1e-30)

def gaussian_kernel(x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    D = my_cdist(x, y)
    K = torch.zeros_like(D)
    for g in gamma:
        K.add_(torch.exp(D.mul(-g)))
    return K

def mmd(x, y):
    Kxx = gaussian_kernel(x, x).mean()
    Kyy = gaussian_kernel(y, y).mean()
    Kxy = gaussian_kernel(x, y).mean()
    return Kxx + Kyy - 2 * Kxy

class Experiment:
    data_config = {
        'train': {'dataset': 'BalanceGroupsDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--mmd_gamma', type=float, default=1.0)
        return ['mmd_gamma']

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = ResNet50()
        self.model.to(DEVICE)
        self.model.train()

        # Optimizer setup
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)

        self.iteration = 0
        self.best_metric = None

    def save(self, path):
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

    def train_iteration(self, data):
        images, targets, t_img, t_targ = data
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        t_img, t_targ = t_img.to(DEVICE), t_targ.to(DEVICE)

        output0, features0 = self.model(images)
        output1, features1 = self.model(t_img)

        objective = torch.cat([
            F.cross_entropy(output0, targets, reduction='none'), 
            F.cross_entropy(output1, t_targ, reduction='none')]).mean()
        penalty = mmd(features0, features1)
            
        loss = (objective + (self.args.mmd_gamma * penalty))

        return {'loss': loss.item()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        predicted = []
        target = []
        group = []

        cls_loss = 0

        for x, y, g in loader:
            x, y, g = x.to(DEVICE), y.to(DEVICE), g.to(DEVICE)

            pred, _ = self.model(x)

            cls_loss += F.cross_entropy(pred, y).item()

            predicted.append(pred)
            target.append(y)
            group.append(g)
        
        predicted = torch.cat(predicted)
        target = torch.cat(target)
        group = torch.cat(group)

        self.model.train()
        
        return predicted, target, group, {'classification_loss': cls_loss / predicted.size(0)}
