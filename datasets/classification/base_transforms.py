import torchvision.transforms as T
from RandAugment import RandAugment

class BaseTrainTransform:
    def __init__(self):
        pass

    def build_transform(self):
        return T.Compose([
                    T.CenterCrop(128),
                    T.Resize(256),
                    RandAugment(3, 15),
                    T.Resize(128),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
    
class BaseTestTransform:
    def __init__(self):
        pass

    def build_transform(self):
        return T.Compose([
                    T.CenterCrop(128),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])