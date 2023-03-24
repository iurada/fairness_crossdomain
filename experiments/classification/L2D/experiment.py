import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50_L2D import ResNet50_L2D
from experiments.classification.L2D.augnet import AugNet
from experiments.classification.L2D.contrastive_loss import SupConLoss
from experiments.classification.L2D.utility import loglikeli, club, conditional_mmd_rbf, l2d_normalize

class Experiment:
    data_config = {
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--l2d_lr_sc', type=float, default=0.005)
        parser.add_argument('--l2d_alpha1', type=float, default=1.0)
        parser.add_argument('--l2d_alpha2', type=float, default=1.0)
        parser.add_argument('--l2d_beta', type=float, default=0.1)
        return ['l2d_lr_sc', 'l2d_alpha1', 'l2d_alpha2', 'l2d_beta']

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = ResNet50_L2D(n_classes=2, pretrained=True)
        self.model.to(DEVICE)
        self.model.train()

        self.convertor = AugNet(1).to(DEVICE)
        self.convertor.train()
        self.convertor_opt = SGD(self.convertor.parameters(), lr=args.l2d_lr_sc)

        self.con = SupConLoss()
        self.tran = l2d_normalize

        self.alpha1 = args.l2d_alpha1
        self.alpha2 = args.l2d_alpha2
        self.beta = args.l2d_beta

        # Optimizer setup
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)

        self.iteration = 0
        self.best_metric = None

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'convertor': self.convertor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'convertor_opt': self.convertor_opt.state_dict(),
            'iteration': self.iteration,
            'best_metric': self.best_metric
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.convertor.load_state_dict(checkpoint['convertor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.convertor_opt.load_state_dict(checkpoint['convertor_opt'])
        self.iteration = checkpoint['iteration']
        self.best_metric = checkpoint['best_metric']

    def train_iteration(self, data):
        x, y, g = data
        x, y = x.to(DEVICE), y.to(DEVICE)

        images = x
        targets = y

        # STAGE 1
        # Aug
        inputs_max = self.tran(torch.sigmoid(self.convertor(images)))
        inputs_max = inputs_max * 0.6 + images * 0.4
        data_aug = torch.cat([inputs_max, images])
        labels = torch.cat([targets, targets])

        # Forward
        logits, logvar, mu, embeddings = self.model(data_aug)

        # Maximize MI between z and z_hat
        emb_src = F.normalize(embeddings[:targets.size(0)]).unsqueeze(1)
        emb_aug = F.normalize(embeddings[targets.size(0):]).unsqueeze(1)
        con = self.con(torch.cat([emb_src, emb_aug], dim=1), targets)

        # Likelihood
        mu_ = mu[targets.size(0):]
        logvar_ = logvar[targets.size(0):]
        y_samples = embeddings[:targets.size(0)]
        likeli = -loglikeli(mu_, logvar_, y_samples)

        # Total loss & backward
        class_loss = F.cross_entropy(logits.squeeze(), labels)
        loss = class_loss + self.alpha2 * likeli + self.alpha1 * con
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # STAGE 2
        inputs_max = self.tran(torch.sigmoid(self.convertor(images, estimation=True)))
        inputs_max = inputs_max * 0.6 + images * 0.4
        data_aug = torch.cat([inputs_max, images])

        # forward with the adapted parameters
        outputs, logvar2, mu2, embeddings2 = self.model(data_aug)

        # Upper bound MI
        mu_ = mu2[targets.size(0):]
        logvar_ = logvar2[targets.size(0):]
        y_samples = embeddings2[:targets.size(0)]
        div = club(mu_, logvar_, y_samples)

        # Semantic consistency
        e = embeddings2
        e1 = e[:targets.size(0)]
        e2 = e[targets.size(0):]
        dist = conditional_mmd_rbf(e1, e2, targets, num_class=2)

        # Total loss and backward
        convertor_loss = dist + self.beta * div
        self.convertor_opt.zero_grad()
        convertor_loss.backward()
        self.convertor_opt.step()

        return {'train_loss': loss.item(), 'convertor_loss': convertor_loss.item()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        predicted = []
        target = []
        group = []

        cls_loss = 0

        for x, y, g in loader:
            x, y, g = x.to(DEVICE), y.to(DEVICE), g.to(DEVICE)

            pred, _, _, _ = self.model(x)

            cls_loss += F.cross_entropy(pred, y).item()

            predicted.append(pred)
            target.append(y)
            group.append(g)
        
        predicted = torch.cat(predicted)
        target = torch.cat(target)
        group = torch.cat(group)

        self.model.train()
        
        return predicted, target, group, {'classification_loss': cls_loss / predicted.size(0)}
