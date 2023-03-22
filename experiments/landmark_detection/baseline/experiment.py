import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from main import DEVICE
from models.landmark_detection.ResNet18 import pose_resnet18
from experiments.landmark_detection.losses import JointsMSELoss

class Experiment:
    data_config = {
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = pose_resnet18(args.image_size, args.num_keypoints, pretrained_backbone=True)
        self.model.to(DEVICE)
        self.model.train()

        self.criterion = JointsMSELoss()

        # Optimizer setup
        self.optimizer = SGD(self.model.get_parameters(lr=0.1), momentum=0.9, weight_decay=0.0001, nesterov=True)
        self.scheduler = MultiStepLR(self.optimizer, [45, 60], gamma=0.1)

        self.iteration = 0
        self.best_metric = None

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'iteration': self.iteration,
            'best_metric': self.best_metric
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.iteration = checkpoint['iteration']
        self.best_metric = checkpoint['best_metric']

    def train_iteration(self, data):
        if self.iteration % self.args.validate_every == 0 and self.iteration > 0:
            self.scheduler.step()

        img, targ, targ_weight, group, lms = data
        img, targ, targ_weight = img.to(DEVICE), targ.to(DEVICE), targ_weight.to(DEVICE)

        pred = self.model(img)
        loss = self.criterion(pred, targ, targ_weight)

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

        loss = 0

        for img, targ, targ_weight, group, lms in loader:
            img, targ, targ_weight = img.to(DEVICE), targ.to(DEVICE), targ_weight.to(DEVICE)

            pred = self.model(img)

            loss += self.criterion(pred, targ).item()

            predicted.append(pred)
            target.append(lms)
            group.append(group)
        
        predicted = torch.cat(predicted)
        target = torch.cat(target)
        group = torch.cat(group)

        self.model.train()
        
        return predicted, target, group, {'loss': loss / predicted.size(0)}
