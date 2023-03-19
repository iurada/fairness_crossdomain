import argparse
import os
import sys
import math

import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

import time
import logging

class FairSupConResNet50(nn.Module):
    """backbone + projection head + classifier"""
    def __init__(self, n_classes=2, 
                    pretrained=True, 
                    hidden_size=2048, 
                    dropout=0.5, 
                    head='mlp', 
                    feat_dim=128
                ):
        super(FairSupConResNet50, self).__init__()

        self.encoder = torchvision.models.resnet50(pretrained=pretrained)
        self.encoder.fc = nn.Linear(2048, hidden_size)

        if head == 'linear':
            self.head = nn.Linear(hidden_size, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def forward_contrastive(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    
    def forward(self, x):
        feat = self.encoder(x)
        out = self.classifier(feat)
        return out, feat

class FairSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(FairSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    
    def forward(self, features, labels, sensitive_labels, group_norm,method, epoch, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: target classes of shape [bsz].
            sensitive_labels: sensitive attributes of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            sensitive_labels = sensitive_labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            sensitive_mask = torch.eq(sensitive_labels, sensitive_labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        sensitive_mask = sensitive_mask.repeat(anchor_count, contrast_count)
        n_sensitive_mask=(~sensitive_mask.bool()).float()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # compute log_prob
        if method=="FSCL":
            mask = mask * logits_mask
            logits_mask_fair=logits_mask*(~mask.bool()).float()*sensitive_mask
            exp_logits_fair = torch.exp(logits) * logits_mask_fair
            exp_logits_sum=exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))

        elif method=="SupCon":
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        elif method=="FSCL*":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask=mask*logits_mask
                
            logits_mask_fair=logits_mask*sensitive_mask
            exp_logits_fair = torch.exp(logits) * logits_mask_fair
            exp_logits_sum=exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))

        elif method=="SimCLR":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask=mask*logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

              
        # compute mean of log-likelihood over positive
        #apply group normalization
        if group_norm==1:
            mean_log_prob_pos = ((mask*log_prob)/((mask*sensitive_mask).sum(1))).sum(1)
           
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        #apply group normalization
        if group_norm==1:
            C=loss.size(0)/8
            norm=(1/(((mask*sensitive_mask).sum(1)+1).float()))
            loss=(loss*norm)*C
            
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

def parse_option(main_opt):
    parser = argparse.ArgumentParser('argument for training')

    opt = dotdict({})
    opt['print_freq'] = 250
    opt['save_freq'] = 1
    opt['epochs'] = main_opt['fscl_pretrain_epochs']

    # optimization
    opt['learning_rate'] = 0.1
    opt['lr_decay_epochs'] = '700,800,900'
    opt['lr_decay_rate'] = 0.1
    opt['weight_decay'] = 1e-4
    opt['momentum'] = 0.9

    opt['method'] = 'FSCL'
    opt['group_norm'] = 1 if main_opt['fscl_group_norm'] else 0
    opt['temp'] = 0.1 # temperature for loss function

    opt['ckpt'] = main_opt['fscl_pretrain_ckpt']

    opt.model_path = os.path.join(main_opt['log_path'], 'FairSupCon')
  
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_lr_{}_decay_{}_temp_{}'.\
        format(opt.method, opt.learning_rate, 
                opt.weight_decay, opt.temp)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)

    return opt

def set_model(opt):
    model = FairSupConResNet50()
    criterion = FairSupConLoss(temperature=opt.temp)
    s_epoch=0

    if opt.ckpt!='':
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        
        if torch.cuda.is_available():
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
            model = model.cuda()
            criterion = criterion.cuda()

            model.load_state_dict(state_dict)
            s_epoch=ckpt['epoch']
    else:

        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()

    return model, criterion,s_epoch

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()

    for idx, (images, ta, sa) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda()
            ta = ta.cuda()
            sa = sa.cuda()
        bsz = ta.shape[0]

        # compute loss
        
        features = model.forward_contrastive(images)
        
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features,ta,sa,opt.group_norm,opt.method,epoch)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            logging.info('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
    
    return losses.avg

def set_optimizer(opt, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return optimizer

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_model(model, optimizer, opt, epoch, save_file):
    logging.info('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def FSCL_pretraining(main_args, train_loader):
    main_opt = vars(main_args)
    opt = parse_option(main_opt)

    # build model and criterion
    s_epoch=0
    model, criterion,s_epoch = set_model(opt)
    
    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(s_epoch+1, opt.epochs + 1):
        
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        logging.info('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'last.pth')
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    return save_file