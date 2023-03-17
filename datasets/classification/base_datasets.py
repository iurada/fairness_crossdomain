from torch.utils.data import Dataset

class BaseDataset(Dataset):

    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        x, y, g = self.examples[index]
        x = self.transform(x)
        return x, y, g