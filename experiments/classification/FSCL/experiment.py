import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from experiments.classification.FSCL.fscl import FairSupConResNet50, FSCL_pretraining

class Experiment:
    data_config = {
        'pretrain': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'FSCLTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': True},
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--fscl_group_norm', action='store_true', help='If set, use FSCL+ instead of default FSCL')
        parser.add_argument('--fscl_pretrain_epochs', type=int, default=280)
        parser.add_argument('--fscl_pretrain_ckpt', type=str, default='')
        parser.add_argument('--fscl_skip_pretrain', action='store_true')
        return ['fscl_group_norm', 'fscl_pretrain_epochs']

    def __init__(self, args, dataloaders):
        self.args = args

        if not args.fscl_skip_pretrain:
            pretrain_ckpt_path = FSCL_pretraining(args, dataloaders['pretrain'])
            args.fscl_pretrain_ckpt = pretrain_ckpt_path
        
        assert os.path.exists(args.fscl_pretrain_ckpt), '[FSCL] Pre-Trained model not found!'

        # Model setup
        self.model = FairSupConResNet50()
        self.model.load_state_dict(torch.load(args.fscl_pretrain_ckpt)['model'])
        self.model.to(DEVICE)
        self.model.eval()

        self.model.classifier.train()

        # Optimizer setup
        self.optimizer = Adam(self.model.classifier.parameters(), lr=1e-4)

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
        x, y, g = data
        images, targets = x.to(DEVICE), y.to(DEVICE)

        with torch.no_grad():
            features = self.model.encoder(images)
        outputs = self.model.classifier(features)
        loss = F.cross_entropy(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.classifier.eval()

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

        self.model.classifier.train()
        
        return predicted, target, group, {'classification_loss': cls_loss / predicted.size(0)}
