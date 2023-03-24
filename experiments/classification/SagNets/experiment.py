import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50_SagNets import SagResNet50

class Experiment:
    data_config = {
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--sagnet_lambda', type=float, default=0.5)
        return ['sagnet_lambda']

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = SagResNet50()
        self.model.to(DEVICE)
        self.model.train()

        # Optimizer setup
        self.optimizer_f = Adam(self.model.network_f.parameters(), lr=1e-4)
        self.optimizer_c = Adam(self.model.network_c.parameters(), lr=1e-4)
        self.optimizer_s = Adam(self.model.network_s.parameters(), lr=1e-4)
        self.lambda_adv = args.sagnet_lambda # Adversarial weight, Hyperparam

        self.iteration = 0
        self.best_metric = None

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer_f': self.optimizer_f.state_dict(),
            'optimizer_c': self.optimizer_c.state_dict(),
            'optimizer_s': self.optimizer_s.state_dict(),
            'iteration': self.iteration,
            'best_metric': self.best_metric
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer_f.load_state_dict(checkpoint['optimizer_f'])
        self.optimizer_c.load_state_dict(checkpoint['optimizer_c'])
        self.optimizer_s.load_state_dict(checkpoint['optimizer_s'])
        self.iteration = checkpoint['iteration']
        self.best_metric = checkpoint['best_metric']

    def train_iteration(self, data):
        x, y, g = data
        x, y = x.to(DEVICE), y.to(DEVICE)

        images = x
        targets = y

        # Content-biased learning
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()

        z = self.model.network_f(images)
        zc = self.model.randomize(z, 'style') # SR

        loss_c = F.cross_entropy(self.model.network_c(zc).squeeze(), targets)
        loss_c.backward()

        self.optimizer_f.step()
        self.optimizer_c.step()

        # Style-biased learning
        self.optimizer_s.zero_grad()

        z = self.model.network_f(images)
        zs = self.model.randomize(z, 'content') # CR

        loss_s = F.cross_entropy(self.model.network_s(zs).squeeze(), targets)
        loss_s.backward()

        self.optimizer_s.step()

        # Adversarial step
        self.optimizer_f.zero_grad()

        z = self.model.network_f(images)
        zs = self.model.randomize(z, 'content') # CR

        loss_adv = (- torch.log(self.model.network_s(zs).softmax(dim=1) + 1e-5).mean(dim=1)).mean()
        loss_adv *= self.lambda_adv
        loss_adv.backward()

        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(), 'loss_adv': loss_adv.item()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        predicted = []
        target = []
        group = []

        cls_loss = 0

        for x, y, g in loader:
            x, y, g = x.to(DEVICE), y.to(DEVICE), g.to(DEVICE)

            pred = self.model(x)

            cls_loss += F.cross_entropy(pred, y).item()

            predicted.append(pred)
            target.append(y)
            group.append(g)
        
        predicted = torch.cat(predicted)
        target = torch.cat(target)
        group = torch.cat(group)

        self.model.train()
        
        return predicted, target, group, {'classification_loss': cls_loss / predicted.size(0)}
