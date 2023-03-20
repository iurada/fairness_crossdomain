import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import logging
import pickle
import random
from main import DEVICE
from models.classification.ResNet50 import ResNet50
from experiments.classification.g_SMOTE.HyperInverter.configs import paths_config
from experiments.classification.g_SMOTE.HyperInverter.models.stylegan2_ada import Generator
from experiments.classification.g_SMOTE.HyperInverter.evaluation.latent_creators import HyperInverterLatentCreator
from experiments.classification.g_SMOTE.HyperInverter.utils.common import tensor2im

class IndexDS(Dataset):
    def __init__(self, train_list):
        self.train_list = train_list
        self.inversion_pre_process = T.Compose(
            [
                T.Resize((1024, 1024)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    
    def __len__(self):
        return len(self.train_list)
    
    def __getitem__(self, index):
        name, attr, prot_attr = self.train_list[index]
        x = self.inversion_pre_process(Image.open(name).convert('RGB'))
        y = torch.tensor([attr, prot_attr])
        return x, y, name

def build_index(train_list, latent_creator):
    index = {0: {0: [], 1: []}, 1: {0: [], 1: []}}
    encoded = {}
    logging.info('[INDEX] Building...')
    db = DataLoader(IndexDS(train_list), batch_size=64, num_workers=4, shuffle=False)
    cnt = 0
    with torch.no_grad():
        for imgs, ys, names in db:
            imgs = imgs.cuda()
            w_encs = latent_creator.embed(imgs).cpu()

            for i in range(len(names)):
                name = names[i]
                attr = ys[i, 0].item()
                prot_attr = ys[i, 1].item()
                index[prot_attr][attr].append(name)
                encoded[name] = w_encs[i].unsqueeze(0)

                if cnt % 500 == 0:
                    logging.info(f'[INDEX] {cnt}/{len(train_list)}')
                cnt += 1
    logging.info('[INDEX] Build complete')
    return index, encoded

def uniform_sample_simplex(simplex_vertices):
    '''
    input: list of vertices of the simplex
    returns: a point uniformely sampled from the convex hull
    '''
    rho_ip1 = simplex_vertices[0]
    for i in range(1, len(simplex_vertices)):
        lambd = torch.rand((1,))**(1/i)
        rho_ip1 = lambd * rho_ip1 + (1 - lambd) * simplex_vertices[i]
    return rho_ip1

def GAN_based_SMOTE(encoded_basepoint, encoded_train_set, m=5, k=3):
    '''
    returns: a latent vector that can be used to generate an image via InvGAN's Generator
    '''
    # Encode train dataset (2 separate encodings based on y)
    # at the beginning, using InvGAN's Discriminator
        
    dist = torch.norm(encoded_basepoint[:, 0, :] - encoded_train_set[:, 0, :], dim=1, p=None)
    top_m_nn = encoded_train_set[dist.topk(m+1, largest=False).indices[1:]]
    idx_k = torch.randperm(m)[:k]
    rand_k_nn = top_m_nn[idx_k]

    return uniform_sample_simplex(rand_k_nn).unsqueeze(0)

class Experiment:
    data_config = {
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--aug_path', type=str, help='Where augmented images get stored.', required=True)
        parser.add_argument('--gen_size', type=int, default=128) # (N x N) output image size
        parser.add_argument('--adaptive', type=bool, default=True)
        parser.add_argument('--lambda', type=float, default=0.5)
        parser.add_argument('--m', type=int, default=5)
        parser.add_argument('--k', type=int, default=3)
        return ['lambda', 'm', 'k']

    def __init__(self, args, dataloaders):
        self.args = args
        self.dataloaders = dataloaders

        # HyperInverter model setup
        model_path = paths_config.model_paths['stylegan2_ada_ffhq']
        with open(model_path, 'rb') as f:
            G_ckpt = pickle.load(f)['G_ema']
            G_ckpt = G_ckpt.float()
        self.G = Generator(**G_ckpt.init_kwargs)
        self.G.load_state_dict(G_ckpt.state_dict())
        self.G.to(DEVICE)
        self.G.eval()

        self.latent_creator = HyperInverterLatentCreator(domain='human_faces')

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

        pred, _ = self.model(x)

        loss = F.cross_entropy(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'train_loss': loss.item()}
    
    @torch.no_grad()
    def generalized_SMOTE(self, predicted, target, group):

        p = torch.argmax(predicted, dim=-1)

        acc_gr0 = (p[group == 0] == target[group == 0]).sum().item() / p[group == 0].size(0)
        acc_gr1 = (p[group == 1] == target[group == 1]).sum().item() / p[group == 1].size(0)

        if self.args.adaptive:
            # Determine weakest group
            if acc_gr0 == acc_gr1:
                return
            elif acc_gr0 < acc_gr1:
                weakest_group = 0
            elif acc_gr0 > acc_gr1:
                weakest_group = 1
        else:
            weakest_group = 0 if random.random() < 0.5 else 1

        # Sample a batch from weakest group
        aug_batch = []
        idxs_already_included0 = []
        idxs_already_included1 = []
        for _ in range(self.args.batch_size):
            attr = 0 if random.random() < 0.5 else 1
            if attr == 0:
                ii = random.choice(list(set(range(len(train_list_index[weakest_group][attr]))) - set(idxs_already_included0)))
                idxs_already_included0.append(ii)
                name = train_list_index[weakest_group][attr][ii]
                aug_batch.append([name, attr, weakest_group])
            else:
                ii = random.choice(list(set(range(len(train_list_index[weakest_group][attr]))) - set(idxs_already_included1)))
                idxs_already_included1.append(ii)
                name = train_list_index[weakest_group][attr][ii]
                aug_batch.append([name, attr, weakest_group])
        
        # Augment the batch
        split0 = []
        split1 = []
        for k in train_list_index[weakest_group][0]:
            split0.append(encoded_trainset[k])
        for k in train_list_index[weakest_group][1]:
            split1.append(encoded_trainset[k])
        split0 = torch.cat(split0)
        split1 = torch.cat(split1)

        for name, attr, prot_attr in aug_batch:
            encoded_basepoint = encoded_trainset[name]
            encoded_split = split0 if attr == 0 else split1
            sample = GAN_based_SMOTE(encoded_basepoint, encoded_split, m=opt['m'], k=opt['k']).to(device)
            generated_img = G.synthesis(sample, added_weights=None, noise_mode="const")[0]
            generated_img = tensor2im(generated_img).resize((opt['gen_size'], opt['gen_size']))
            
            out_name = f"{opt['aug_path']}{opt['attribute']}/{attr}_{prot_attr}_{g_smote_filecount}.png"
            generated_img.save(out_name)
            g_smote_filecount += 1

            aug_list.append([out_name, attr, prot_attr])

        # Rebuild train loader
        train_loader = DataLoader(
            TrainAugDataset(train_list, aug_list, transform_train, opt['lambda']),
            batch_size=opt['batch_size'],
            shuffle=True,
            num_workers=opt['num_workers']
        )

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

        self.generalized_SMOTE(predicted, target, group)
        
        return predicted, target, group, {'classification_loss': cls_loss / predicted.size(0)}
