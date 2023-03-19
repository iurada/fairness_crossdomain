import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50_AFN import ResNet50_AFN

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
        'train': {'dataset': 'BalanceGroupsDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--afn_type', type=str, choices=['HAFN', 'SAFN'], default='HAFN')
        parser.add_argument('--afn_radius', type=float, default=25.0)
        parser.add_argument('--afn_weight_L2norm', type=float, default=0.05)
        return ['afn_type', 'afn_radius', 'afn_weight_L2norm']

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = ResNet50_AFN()
        self.model.to(DEVICE)
        self.model.train()

        # Optimizer setup
        self.opt_g = Adam(self.model.netG.parameters(), lr=1e-4)
        self.opt_f = Adam(self.model.netF.parameters(), lr=1e-4)

        self.iteration = 0
        self.best_metric = None

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'opt_g': self.opt_g.state_dict(),
            'opt_f': self.opt_f.state_dict(),
            'iteration': self.iteration,
            'best_metric': self.best_metric
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.opt_g.load_state_dict(checkpoint['opt_g'])
        self.opt_f.load_state_dict(checkpoint['opt_f'])
        self.iteration = checkpoint['iteration']
        self.best_metric = checkpoint['best_metric']

    def train_iteration(self, data):
        images, targets, t_img, t_targ = data
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        t_img, t_targ = t_img.to(DEVICE), t_targ.to(DEVICE)

        self.opt_g.zero_grad()
        self.opt_f.zero_grad()

        s_bottleneck = self.model.netG(images)
        t_bottleneck = self.model.netG(t_img)
        s_fc2_emb, s_logit = self.model.netF(s_bottleneck)
        t_fc2_emb, t_logit = self.model.netF(t_bottleneck)

        s_cls_loss = F.cross_entropy(s_logit, targets)
        t_cls_loss = F.cross_entropy(t_logit, t_targ)
        s_fc2_L2norm_loss = get_L2norm_loss_self_driven(s_fc2_emb, self.args)
        t_fc2_L2norm_loss = get_L2norm_loss_self_driven(t_fc2_emb, self.args)

        loss = s_cls_loss + t_cls_loss + s_fc2_L2norm_loss + t_fc2_L2norm_loss
        loss.backward()

        self.opt_g.step()
        self.opt_f.step()

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

            _, pred = self.model(x)

            cls_loss += F.cross_entropy(pred, y).item()

            predicted.append(pred)
            target.append(y)
            group.append(g)
        
        predicted = torch.cat(predicted)
        target = torch.cat(target)
        group = torch.cat(group)

        self.model.train()
        
        return predicted, target, group, {'classification_loss': cls_loss / predicted.size(0)}
