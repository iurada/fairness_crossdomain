from torch.utils.data import Dataset
from PIL import Image
import random

class BaseDataset(Dataset):

    def __init__(self, examples, transform):
        # examples is a list of [img_id(path), target(int), group(int)]
        # transform is a torchvision.transforms.Compose(...)
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y, g = self.examples[index]
        x = Image.open(img_path).convert('RGB')
        x = self.transform(x)
        return x, y, g
    
class BalanceGroupsDataset(Dataset):

    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

        group0 = []
        group1 = []

        for example in examples:
            if example[-1] == 0:
                group0.append(example)
            else:
                group1.append(example)

        self.dataset_len = max(len(group0), len(group1))

        if len(group0) > len(group1):
            self.source = group0
            self.target = group1
        else:
            self.source = group1
            self.target = group0

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        s_id, s_targ, _ = self.source[index]
        t_id, t_targ, _ = random.choice(self.target)

        s_img = self.transform(Image.open(s_id))
        t_img = self.transform(Image.open(t_id))

        return s_img, s_targ, t_img, t_targ

class RotationDataset(Dataset):

    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

        try:
            self.interp = Image.Resampling.NEAREST
        except AttributeError:
            self.interp = Image.NEAREST

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        ID, y, g = self.examples[index]
        img = Image.open(ID)
        
        y_rot = random.randint(0, 3)
        img_rot = img.rotate(y_rot * 90, self.interp, expand=True)

        X = self.transform(img)
        X_rot = self.transform(img_rot)
        
        return X, y, X_rot, y_rot
    