import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50_SelfReg import ResNet50_SelfReg
import numpy as np

class Experiment:
    data_config = {
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--selfreg_alpha', type=float, default=0.5)
        parser.add_argument('--selfreg_beta', type=float, default=0.5)
        parser.add_argument('--selfreg_lam_feature', type=float, default=0.3)
        parser.add_argument('--selfreg_lam_logit', type=float, default=1.0)
        return ['selfreg_alpha', 'selfreg_beta', 'selfreg_lam_feature', 'selfreg_lam_logit']

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = ResNet50_SelfReg()
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

        images = x
        targets = y

        lam = np.random.beta(self.args.selfreg_alpha, self.args.selfreg_beta)
        batch_size = x.size(0)

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_targets, indices = torch.sort(targets)
            sorted_images = torch.zeros_like(images)
            for idx, order in enumerate(indices):
                sorted_images[idx] = images[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_targets):
                if ex == val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            images = sorted_images
            targets = sorted_targets

        output, feat = self.model(images)
        proj = self.model.cdpl(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end - ex) + ex
            shuffle_indices2 = torch.randperm(end - ex) + ex
            for idx in range(end - ex):
                output_2[idx + ex] = output[shuffle_indices[idx]]
                feat_2[idx + ex] = proj[shuffle_indices[idx]]
                output_3[idx + ex] = output[shuffle_indices2[idx]]
                feat_3[idx + ex] = proj[shuffle_indices2[idx]]
            ex = end
        
        # mixup
        output_3 = lam * output_2 + (1 - lam) * output_3
        feat_3 = lam * feat_2 + (1 - lam) * feat_3
        
        # regularization
        L_ind_logit = self.args.selfreg_lam_logit * F.mse_loss(output, output_2)
        L_hdl_logit = self.args.selfreg_lam_logit * F.mse_loss(output, output_3)
        L_ind_feat = self.args.selfreg_lam_feature * F.mse_loss(feat, feat_2)
        L_hdl_feat = self.args.selfreg_lam_feature * F.mse_loss(feat, feat_3)

        cl_loss = F.cross_entropy(output, targets)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale * (lam * (L_ind_logit + L_ind_feat) + (1 - lam) * (L_hdl_logit + L_hdl_feat))

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
