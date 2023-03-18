from torch.utils.data import Dataset
from PIL import Image

class BaseDataset(Dataset):

    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y, g = self.examples[index]
        x = Image.open(img_path).convert('RGB')
        x = self.transform(x)
        return x, y, g