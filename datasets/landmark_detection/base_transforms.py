import torchvision.transforms as T
import torchlm as TLM

class BaseTrainTransform:
    def __init__(self):
        pass

    def build_transform(self, args):
        return (TLM.LandmarksCompose([
                    TLM.LandmarksResize(args.image_size),
                    TLM.LandmarksRandomRotate(180),
                    TLM.LandmarksRandomHorizontalFlip(),
                    TLM.LandmarksRandomShear((0.6, 1.3))
            ]), T.Compose([
                    T.ColorJitter(brightness=0.24, contrast=0.25, saturation=0.25),
                    T.GaussianBlur(5, sigma=(0.1, 0.8)),
                    T.ToTensor(), 
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ]))
    
class BaseTestTransform:
    def __init__(self):
        pass

    def build_transform(self, args):
        return (TLM.LandmarksCompose([
                    TLM.LandmarksResize(args.image_size)
            ]), T.Compose([
                    T.ToTensor(), 
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ]))