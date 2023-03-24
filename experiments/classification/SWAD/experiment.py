import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50 import ResNet50
from experiments.classification.SWAD import swad as swad_module
from experiments.classification.SWAD import swa_utils
import logging
#import pickle

class Experiment:
    data_config = {
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--n_converge', type=int, default=3)
        parser.add_argument('--n_tolerance', type=int, default=6)
        parser.add_argument('--tolerance_ratio', type=float, default=0.3)
        parser.add_argument('--start_it', type=float, default=0)
        return ['n_converge', 'n_tolerance', 'tolerance_ratio', 'start_it']

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = ResNet50()
        self.model.to(DEVICE)
        self.model.train()

        self.swad_algorithm = swa_utils.AveragedModel(self.model)
        self.swad = swad_module.LossValley(args.n_converge, args.n_tolerance, 
                                           args.tolerance_ratio, args.start_it)

        # Optimizer setup
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)

        self.iteration = 0
        self.best_metric = None

        self.train_loader = dataloaders['train']

    def save(self, path, can_save_best=False):
        folder_path, file_path = os.path.split(path)
        self.save_folder_path = folder_path

        if file_path == 'best.pth' and can_save_best == False:
            return

        torch.save({
            'model': self.model.state_dict(),
            'swad_algorithm': self.swad_algorithm.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iteration': self.iteration,
            'best_metric': self.best_metric
        }, path)
        #with open(os.path.join(folder_path, 'loss_valley.pkl'), 'wb') as f:
        #    pickle.dump(self.swad, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.swad_algorithm.load_state_dict(checkpoint['swad_algorithm'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iteration = checkpoint['iteration']
        self.best_metric = checkpoint['best_metric']
        #with open(os.path.join(path, 'loss_valley.pkl'), 'rb') as f:
        #    self.swad = pickle.load(f)

    def process_last_iteration(self):
        if (self.iteration + 1) < self.args.max_iters:
            return
        
        self.swad_algorithm = self.swad.get_final_model()
        swa_utils.update_bn(self.train_loader, self.swad_algorithm, 500)
        self.model = self.swad_algorithm.module
        self.save(os.path.join(self.save_folder_path, 'best.pth'), can_save_best=True)

    def train_iteration(self, data):
        x, y, g = data
        x, y = x.to(DEVICE), y.to(DEVICE)

        pred, _ = self.model(x)

        loss = F.cross_entropy(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.swad_algorithm.update_parameters(self.model, step=self.iteration)

        self.process_last_iteration()

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

        self.swad.update_and_evaluate(self.swad_algorithm, _, cls_loss)
        if hasattr(self.swad, 'dead_valley') and self.swad.dead_valley:
            logging.info('SWAD valley is dead -> early stop!')
            self.iteration = self.args.max_iters
            self.process_last_iteration()
        else:
            self.swad_algorithm = swa_utils.AveragedModel(self.model)

        return predicted, target, group, {'classification_loss': cls_loss / predicted.size(0)}
