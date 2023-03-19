import torch
from torch.optim import Adam
import torch.nn.functional as F
from main import DEVICE
from models.classification.ResNet50_DANN import ResNet50_DANN
import torch.autograd as autograd

class Experiment:
    data_config = {
        'train': {'dataset': 'BaseDataset', 'set': 'train_set', 'transform': 'BaseTrainTransform', 'filter': None, 'shuffle': True,  'drop_last': False},
        'val':   {'dataset': 'BaseDataset', 'set': 'val_set',   'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False},
        'test':  {'dataset': 'BaseDataset', 'set': 'test_set',  'transform': 'BaseTestTransform',  'filter': None, 'shuffle': False, 'drop_last': False}
    }

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--dann_grad_penalty', type=float, default=0.01)
        parser.add_argument('--dann_d_steps_per_g_step', type=int, default=1)
        parser.add_argument('--dann_lambda', type=float, default=1.0)
        return ['dann_grad_penalty', 'dann_d_steps_per_g_step', 'dann_lambda']

    def __init__(self, args, dataloaders):
        self.args = args

        # Model setup
        self.model = ResNet50_DANN()
        self.model.to(DEVICE)
        self.model.train()

        # Optimizer setup
        self.disc_opt = Adam(
            list(self.model.discriminator.parameters()) + list(self.model.class_embeddings.parameters()),
            lr=1e-4)

        self.gen_opt = Adam(
            list(self.model.featurizer.parameters()) + list(self.model.classifier.parameters()),
            lr=1e-4)

        self.iteration = 0
        self.best_metric = None

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'disc_opt': self.disc_opt.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'iteration': self.iteration,
            'best_metric': self.best_metric
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.disc_opt.load_state_dict(checkpoint['disc_opt'])
        self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        self.iteration = checkpoint['iteration']
        self.best_metric = checkpoint['best_metric']

    def train_iteration(self, data):
        x, y, g = data
        x, y, g = x.to(DEVICE), y.to(DEVICE), g.to(DEVICE)

        images = x
        disc_labels = g
        self.model.update_count += 1

        z = self.model.featurizer(images)
        disc_input = z
        
        disc_out = self.model.discriminator(disc_input)
        
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        
        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(), 
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.args.dann_grad_penalty * grad_penalty

        d_steps_per_g = self.args.dann_d_steps_per_g_step
        if (self.model.update_count % (1 + d_steps_per_g) < d_steps_per_g):
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.model.classifier(z)
            classifier_loss = F.cross_entropy(all_preds, y)
            gen_loss = (classifier_loss + (self.args.dann_lambda * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        predicted = []
        target = []
        group = []

        cls_loss = 0

        for x, y, g in loader:
            x, y, g = x.to(DEVICE), y.to(DEVICE), g.to(DEVICE)

            pred = self.model(x)

            cls_loss += F.cross_entropy(pred, y).item()

            predicted.append(pred)
            target.append(y)
            group.append(g)
        
        predicted = torch.cat(predicted)
        target = torch.cat(target)
        group = torch.cat(group)

        self.model.train()
        
        return predicted, target, group, {'classification_loss': cls_loss / predicted.size(0)}
