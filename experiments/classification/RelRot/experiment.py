import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50_Rotation import ResNet50_Rotation

class Experiment:
    data_config = {
        'train': {'dataset': 'RotationDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--rotation_alpha', type=float, default=1.0)
        return ['rotation_alpha']

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = ResNet50_Rotation()
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
        images, targets, images_rot, targets_rot = data
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        images_rot, targets_rot = images_rot.to(DEVICE), targets_rot.to(DEVICE)

        outputs, outputs_rot = self.model(images, images_rot)

        loss_cls = F.cross_entropy(outputs.squeeze(), targets)
        loss_rot = F.cross_entropy(outputs_rot.squeeze(), targets_rot)
        loss = loss_cls + self.args.rotation_alpha * loss_rot

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss_cls': loss_cls.item(), 'loss_rot': loss_rot.item()}

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
