import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50 import ResNet50

class Experiment:
    data_config = {
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--dro_eta_q', type=float, default=0.01)
        parser.add_argument('--dro_c', type=float, default=1.0)
        return ['dro_eta_q', 'dro_c']

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = ResNet50()
        self.model.to(DEVICE)
        self.model.train()

        # Optimizer setup
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)

        self.n_groups = 2
        n_groups_count = [0 for _ in range(self.n_groups)]
        for example in dataloaders['train'].dataset.examples:
            n_groups_count[example[-1]] += 1
        
        self.q = None
        self.adj = args.dro_c * torch.ones(self.n_groups, device=DEVICE) / torch.sqrt(torch.tensor(n_groups_count, device=DEVICE, dtype=torch.float32))
        self.eta_q = args.dro_eta_q * torch.ones(self.n_groups, device=DEVICE)

        self.iteration = 0
        self.best_metric = None

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'q': self.q.clone() if self.q is not None else None,
            'iteration': self.iteration,
            'best_metric': self.best_metric
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.q = checkpoint['q'].clone() if checkpoint['q'] is not None else None
        self.iteration = checkpoint['iteration']
        self.best_metric = checkpoint['best_metric']

    def train_iteration(self, data):
        x, y, g = data
        images, targets, group_labels = x.to(DEVICE), y.to(DEVICE), g.to(DEVICE)

        outputs, _ = self.model(images)
        outputs.squeeze()

        if self.q is None:
            q = torch.ones(self.n_groups, device=DEVICE) / self.n_groups
        else:
            q = self.q.clone()

        loss_0 = F.cross_entropy(outputs[group_labels == 0], targets[group_labels == 0]) # loss for group 0
        loss_1 = F.cross_entropy(outputs[group_labels == 1], targets[group_labels == 1]) # loss for group 1

        losses = torch.stack([loss_0, loss_1])

        # Compute Adjusted losses
        losses += self.adj

        q *= (self.eta_q * losses).exp()
        q /= q.sum()

        self.q = q.clone().detach()
        loss = (q * losses).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
