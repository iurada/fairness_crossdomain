import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50 import ResNet50
import copy
from collections import OrderedDict
from numbers import Number
import operator

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)

class Experiment:
    data_config = {
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--fish_meta_lr', type=float, default=0.5, choices=[0.05, 0.1, 0.5])
        return ['fish_meta_lr']

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = ResNet50()
        self.model.to(DEVICE)
        self.model.train()

        # Optimizer setup
        self.optim_inner_state = None

        self.iteration = 0
        self.best_metric = None

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optim_inner_state': copy.deepcopy(self.optim_inner_state),
            'iteration': self.iteration,
            'best_metric': self.best_metric
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optim_inner_state = copy.deepcopy(checkpoint['optim_inner_state'])
        self.iteration = checkpoint['iteration']
        self.best_metric = checkpoint['best_metric']

    def train_iteration(self, data):
        x, y, g = data

        images = x.to(DEVICE)
        targets = y.to(DEVICE)

        # Create a clone of the network
        model_inner = ResNet50()
        model_inner.load_state_dict(copy.deepcopy(self.model.state_dict()))
        model_inner.require_all_grads()
        model_inner.train()
        model_inner.to(DEVICE)

        optim_inner = Adam(model_inner.parameters(), lr=1e-4)
        if self.optim_inner_state is not None:
            optim_inner.load_state_dict(self.optim_inner_state)

        # Update weights
        outputs, _ = model_inner(images)
        loss = F.cross_entropy(outputs, targets)
        optim_inner.zero_grad()
        loss.backward()
        optim_inner.step()

        self.optim_inner_state = copy.deepcopy(optim_inner.state_dict())
        meta_weights = ParamDict(self.model.state_dict())
        inner_weights = ParamDict(model_inner.state_dict())
        meta_weights += self.args.fish_meta_lr * (inner_weights - meta_weights)

        # Reset weights
        self.model.load_state_dict(copy.deepcopy(meta_weights))

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
