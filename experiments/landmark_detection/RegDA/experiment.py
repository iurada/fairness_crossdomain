import os
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import torch.nn.functional as F
from main import DEVICE
from models.landmark_detection.ResNet18_RegDA import PoseResNet2d as RegDAPoseResNet, \
    PseudoLabelGenerator2d, RegressionDisparity
from models.landmark_detection.ResNet18 import resnet18, Upsampling, PoseResNet
from experiments.landmark_detection.losses import JointsKLLoss
from metrics.landmark_detection.meters import SDR
from metrics.utils import collect_metrics
import logging


def RegDA_pretraining(args, train_loader, val_loader):
    # opt.model_path = os.path.join(main_opt['log_path'], 'FairSupCon')
    # opt.model_name = '{}_lr_{}_decay_{}_temp_{}'.\
    #     format(opt.method, opt.learning_rate, 
    #             opt.weight_decay, opt.temp)
    # opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    # if not os.path.isdir(opt.save_folder):
    #     os.makedirs(opt.save_folder, exist_ok=True)

    backbone = resnet18(pretrained=True)
    upsampling = Upsampling(backbone.out_features)
    model = PoseResNet(backbone, upsampling, args.image_size, args.num_keypoints, True).to(DEVICE)
    model.train()
    optimizer = SGD(model.get_parameters(lr=0.1), momentum=0.9, weight_decay=0.0001, nesterov=True)
    lr_scheduler = MultiStepLR(optimizer, [45, 60], gamma=0.1)
    criterion = JointsKLLoss()

    meters_dict = {}
    meters_dict['SDR'] = SDR(args, meters_dict)

    iteration = 0
    best_metric = None

    pretrain_checkpoint_path = os.path.join(args.log_path, 'pretrain.pth')

    while iteration < args.regda_pretrain_iters:

        for s_img, s_targ, s_targ_weight, _, _ in train_loader:
            
            if iteration % args.validate_every == 0 and iteration > 0:
                lr_scheduler.step()

            optimizer.zero_grad()

            x_s, label_s, weight_s = s_img.to(DEVICE), s_targ.to(DEVICE), s_targ_weight.to(DEVICE)

            # compute output
            y_s = model(x_s)
            loss_s = criterion(y_s, label_s, weight_s)

            # compute gradient and do SGD step
            loss_s.backward()
            optimizer.step()

            if iteration % args.validate_every == 0:
                model.eval()

                predicted = []
                target = []
                group = []

                loss = 0

                for img, targ, targ_weight, group, lms in val_loader:
                    img, targ, targ_weight = img.to(DEVICE), targ.to(DEVICE), targ_weight.to(DEVICE)

                    pred = model(img)
                    loss += criterion(pred, targ, targ_weight).item()

                    predicted.append(pred * args.image_size / args.heatmap_size)
                    target.append(lms)
                    group.append(group)
                
                predicted = torch.cat(predicted)
                target = torch.cat(target)
                group = torch.cat(group)

                model.train()

                metrics = collect_metrics(meters_dict, predicted, target, group) 

                # Log metrics eg. via wandb
                logging.info(f'[PRETRAIN VAL @ {iteration}] {loss / predicted.size(0)} | {metrics}')

                if best_metric is None or meters_dict['SDR'].compare(metrics['SDR'], best_metric):
                    best_metric = metrics['SDR']
                    torch.save({
                        'model': model.state_dict()
                    }, pretrain_checkpoint_path)

        iteration += 1
        if iteration >= args.regda_pretrain_iters: break
    
    return pretrain_checkpoint_path

def filter_keep_gr0(examples):
    return [example for example in examples if example[-1] == 0]

class Experiment:
    data_config = {
        'pretrain': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': filter_keep_gr0, 'shuffle': True,  'drop_last': False},
        'pretrain_val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': filter_keep_gr0, 'shuffle': False, 'drop_last': False},
        'train': {'dataset': 'BalanceGroupsDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--regda_margin', type=float, default=4.0)
        parser.add_argument('--regda_tradeoff', type=float, default=1.0)
        parser.add_argument('--regda_pretrain_iters', type=int, default=35000)
        parser.add_argument('--regda_pretrain_ckpt', type=str, default='')
        parser.add_argument('--regda_skip_pretrain', action='store_true')
        return ['regda_margin', 'regda_tradeoff', 'regda_pretrain_iters']

    def __init__(self, args, dataloaders):
        self.args = args

        if not args.regda_skip_pretrain or not args.test_mode:
            pretrain_ckpt_path = RegDA_pretraining(args, dataloaders['pretrain'], dataloaders['pretrain_val'])
            args.regda_pretrain_ckpt = pretrain_ckpt_path
        
        assert os.path.exists(args.regda_pretrain_ckpt), '[RegDA] Pre-Trained model not found!'

        # Model setup
        backbone = resnet18(pretrained=True)
        upsampling = Upsampling(backbone.out_features)
        self.model = RegDAPoseResNet(backbone, upsampling, args.image_size, args.num_keypoints, 2, finetune=True)
        
        pretrained_dict = torch.load(args.regda_pretrain_ckpt, map_location='cpu')['model']
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        self.model.load_state_dict(pretrained_dict, strict=False)

        self.model.to(DEVICE)
        self.model.train()

        self.criterion = JointsKLLoss()
        pseudo_label_generator = PseudoLabelGenerator2d(args.num_keypoints, args.heatmap_size, args.heatmap_size)
        self.regression_disparity = RegressionDisparity(pseudo_label_generator, JointsKLLoss(epsilon=1e-7))

        # Optimizer setup
        self.optimizer_f = SGD([
            {'params': backbone.parameters(), 'lr': 0.1},
            {'params': upsampling.parameters(), 'lr': 0.1},
        ], lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
        self.optimizer_h = SGD(self.model.head.parameters(), lr=1., momentum=0.9, weight_decay=0.0001, nesterov=True)
        self.optimizer_h_adv = SGD(self.model.head_adv.parameters(), lr=1., momentum=0.9, weight_decay=0.0001, nesterov=True)
        lr_decay_function = lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)
        self.lr_scheduler_f = LambdaLR(self.optimizer_f, lr_decay_function)
        self.lr_scheduler_h = LambdaLR(self.optimizer_h, lr_decay_function)
        self.lr_scheduler_h_adv = LambdaLR(self.optimizer_h_adv, lr_decay_function)
        
        self.iteration = 0
        self.best_metric = None

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer_f': self.optimizer_f.state_dict(),
            'optimizer_h': self.optimizer_h.state_dict(),
            'optimizer_h_adv': self.optimizer_h_adv.state_dict(),
            'lr_scheduler_f': self.lr_scheduler_f.state_dict(),
            'lr_scheduler_h': self.lr_scheduler_h.state_dict(),
            'lr_scheduler_h_adv': self.lr_scheduler_h_adv.state_dict(),
            'iteration': self.iteration,
            'best_metric': self.best_metric
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer_f.load_state_dict(checkpoint['optimizer_f'])
        self.optimizer_h.load_state_dict(checkpoint['optimizer_h'])
        self.optimizer_h_adv.load_state_dict(checkpoint['optimizer_h_adv'])
        self.lr_scheduler_f.load_state_dict(checkpoint['lr_scheduler_f'])
        self.lr_scheduler_h.load_state_dict(checkpoint['lr_scheduler_h'])
        self.lr_scheduler_h_adv.load_state_dict(checkpoint['lr_scheduler_h_adv'])
        self.iteration = checkpoint['iteration']
        self.best_metric = checkpoint['best_metric']

    def train_iteration(self, data):

        s_img, s_targ, s_targ_weight, t_img, t_targ, t_targ_weight = data

        x_s, label_s, weight_s = s_img.to(DEVICE), s_targ.to(DEVICE), s_targ_weight.to(DEVICE)
        x_t, weight_t = t_img.to(DEVICE), t_targ_weight.to(DEVICE)

        # Step A train all networks to minimize loss on source domain
        self.optimizer_f.zero_grad()
        self.optimizer_h.zero_grad()
        self.optimizer_h_adv.zero_grad()

        y_s, y_s_adv = self.model(x_s)
        loss_s = self.criterion(y_s, label_s, weight_s) + \
                 self.args.regda_margin * self.args.regda_tradeoff * self.regression_disparity(y_s, y_s_adv, weight_s, mode='min')
        loss_s.backward()
        self.optimizer_f.step()
        self.optimizer_h.step()
        self.optimizer_h_adv.step()

        # Step B train adv regressor to maximize regression disparity
        self.optimizer_h_adv.zero_grad()
        y_t, y_t_adv = self.model(x_t)
        loss_ground_false = self.args.regda_tradeoff * self.regression_disparity(y_t, y_t_adv, weight_t, mode='max')
        loss_ground_false.backward()
        self.optimizer_h_adv.step()

        # Step C train feature extractor to minimize regression disparity
        self.optimizer_f.zero_grad()
        y_t, y_t_adv = self.model(x_t)
        loss_ground_truth = self.args.regda_tradeoff * self.regression_disparity(y_t, y_t_adv, weight_t, mode='min')
        loss_ground_truth.backward()
        self.optimizer_f.step()

        # do update step
        self.model.step()
        self.lr_scheduler_f.step()
        self.lr_scheduler_h.step()
        self.lr_scheduler_h_adv.step()

        return {'loss_s': loss_s.item(), 'loss_ground_false': loss_ground_false.item(), 'loss_ground_truth': loss_ground_truth.item()}

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
