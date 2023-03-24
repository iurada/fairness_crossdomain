import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from main import DEVICE
from models.landmark_detection.ResNet18 import pose_resnet18_DANN
from experiments.landmark_detection.losses import JointsMSELoss
import math

def get_L2norm_loss_self_driven(x, args):
    if args.afn_type == 'SAFN':
        radius = x.norm(p=2, dim=1).detach()
        assert radius.requires_grad == False
        radius = radius + 1.0 #2.0
        l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
        return args.afn_weight_L2norm * l
    else:
        l = (x.norm(p=2, dim=1).mean() - args.afn_radius) ** 2
        return args.afn_weight_L2norm * l

class Experiment:
    data_config = {
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--dann_lambda', type=float, default=1.0)
        return ['dann_lambda']

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = pose_resnet18_DANN(args.image_size, args.num_keypoints, pretrained_backbone=True)
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

        img, targ, weight = img.to(DEVICE), targ.to(DEVICE), targ_weight.to(DEVICE)

        dom_labels = torch.zeros((img.size(0), 2))
        dom_labels[group == 0, 0] = 1
        dom_labels[group == 1, 1] = 1
        dom_labels = dom_labels.to(DEVICE)

        z = self.model.backbone(img)
        dom_pred = self.model.discriminator(self.model.discriminator.preprocess(z))
        pred = self.model.head(self.model.upsampling(z))

        disc_loss = F.cross_entropy(dom_pred, dom_labels)
        lm_loss = self.criterion(pred, targ, weight)

        loss = lm_loss - self.args.dann_lambda * disc_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'lm_loss': lm_loss.item(), 'disc_loss': disc_loss.item()}

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

            loss += self.criterion(pred, targ, targ_weight).item()

            predicted.append(pred * self.args.image_size / self.args.heatmap_size)
            target.append(lms)
            group.append(group)
        
        predicted = torch.cat(predicted)
        target = torch.cat(target)
        group = torch.cat(group)

        self.model.train()
        
        return predicted, target, group, {'loss': loss / predicted.size(0)}
