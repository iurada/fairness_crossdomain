import torchvision.transforms as T
try:
    from RandAugment import RandAugment
except ModuleNotFoundError:
    from torchvision.transforms import RandAugment

class BaseTrainTransform:
    def __init__(self):
        pass

    def build_transform(self, args):
        return T.Compose([
                    T.CenterCrop(args.image_size),
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

    def build_transform(self, args):
        return T.Compose([
                    T.CenterCrop(args.image_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class FSCLTrainTransform:
    def __init__(self):
        pass

    def build_transform(self, args):
        return TwoCropTransform(T.Compose([
                T.RandomResizedCrop(size=args.image_size, scale=(0.2, 1.)),
                T.RandomHorizontalFlip(),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ]))