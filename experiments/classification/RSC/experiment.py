import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50 import ResNet50
import torch.autograd as autograd
import numpy as np

class Experiment:
    data_config = {
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--rsc_f_drop_factor', type=float, default=0.250)
        parser.add_argument('--rsc_b_drop_factor', type=float, default=0.250)
        return ['rsc_f_drop_factor', 'rsc_b_drop_factor']

    def __init__(self, args, dataloaders):
        self.args = args

        self.drop_f = (1 - args.rsc_f_drop_factor) * 100
        self.drop_b = (1 - args.rsc_b_drop_factor) * 100
        self.num_classes = 2

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
        x, y, g = data
        x, y = x.to(DEVICE), y.to(DEVICE)

        # inputs
        all_x = x
        # labels
        all_y = y
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y.long(), self.num_classes)
        # features

        x = self.model.resnet.conv1(all_x)
        x = self.model.resnet.bn1(x)
        x = self.model.resnet.relu(x)
        x = self.model.resnet.maxpool(x)

        x = self.model.resnet.layer1(x)
        x = self.model.resnet.layer2(x)
        x = self.model.resnet.layer3(x)
        x = self.model.resnet.layer4(x)
        x = self.model.resnet.avgpool(x)
        all_f = torch.flatten(x, 1)

        # predictions

        features = self.model.resnet.fc(all_f)
        all_p = self.model.fc(self.model.relu(features))

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device=DEVICE)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions

        features = self.model.resnet.fc(all_f_muted)
        all_p_muted = self.model.fc(self.model.relu(features))

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples

        features = self.model.resnet.fc(all_f * mask)
        all_p_muted_again = self.model.fc(self.model.relu(features))

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y.long())
        
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
