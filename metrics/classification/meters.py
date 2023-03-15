import torch
from metrics.utils import AbstractMeter

# Remember to sort meters according to the computational order 
# -- This will be the full presentation order:
METRICS_ORDER = ['Acc', 'MGA', 'mGA', 'DA', 'DEO', 'DEOdds', 'DTO', 'DeltaDTO', 'HF']

class Acc(AbstractMeter):
    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None

    def compute(self, predicted, target, group):
        predicted = torch.argmax(predicted, dim=-1)
        self.value = (predicted == target).sum().item() / predicted.size(0)

        acc_gr0 = (predicted[:, 0][group[:, 0] == 0] == target[:, 0][group[:, 0] == 0]).sum().item() / predicted[:, 0][group[:, 0] == 0].size(0)
        acc_gr1 = (predicted[:, 0][group[:, 0] == 1] == target[:, 0][group[:, 0] == 1]).sum().item() / predicted[:, 0][group[:, 0] == 1].size(0)

        if 'MGA' in self.meters_dict.keys():
            self.meters_dict['MGA'].value = max(acc_gr0, acc_gr1)
        if 'mGA' in self.meters_dict.keys():
            self.meters_dict['mGA'].value = min(acc_gr0, acc_gr1)

        return self.value

    @staticmethod
    def compare(current, best):
        return current > best

class MGA(AbstractMeter):
    required_metrics = ['Acc']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
    
    def compute(self, predicted, target, group):
        return self.value
    
    @staticmethod
    def compare(current, best):
        return current > best
    
class mGA(AbstractMeter):
    required_metrics = ['Acc']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None

    def compute(self, predicted, target, group):
        return self.value

    @staticmethod
    def compare(current, best):
        return current > best

class DA(AbstractMeter):
    required_metrics = ['MGA', 'mGA']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None

    def compute(self, predicted, target, group):
        self.value = self.meters_dict['MGA'].value - self.meters_dict['mGA'].value
        return self.value
    
    @staticmethod
    def compare(current, best):
        return current < best

class HF(AbstractMeter):
    required_metrics = ['MGA', 'DA']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
        self.baseline_MGA = args.baseline_MGA
        self.baseline_DA = args.baseline_DA

    def compute(self, predicted, target, group):
        MGA = self.meters_dict['MGA'] * 100
        DA = self.meters_dict['DA'] * 100
        a = (100 + MGA - self.baseline_MGA) / 2
        b = (100 + self.baseline_DA - DA) / 2
        self.value = (2 * a * b) / (a + b)
        return self.value

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--baseline_MGA', type=float, required=True)
        parser.add_argument('--baseline_DA', type=float, required=True)
    
    @staticmethod
    def compare(current, best):
        return current > best

class DTO(AbstractMeter):
    required_metrics = ['MGA', 'mGA']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None

    def compute(self, predicted, target, group):
        MGA = self.meters_dict['MGA'] * 100
        mGA = self.meters_dict['mGA'] * 100
        self.value = ((100 - MGA)**2 + (100 - mGA)**2)**0.5
        return self.value

    @staticmethod
    def compare(current, best):
        return current < best

class DeltaDTO(AbstractMeter):
    required_metrics = ['MGA', 'mGA']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
        self.baseline_DTO = args.baseline_DTO

    def compute(self, predicted, target, group):
        MGA = self.meters_dict['MGA'] * 100
        mGA = self.meters_dict['mGA'] * 100
        self.value = self.baseline_DTO - ((100 - MGA)**2 + (100 - mGA)**2)**0.5
        return self.value

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--baseline_DTO', type=float, required=True)
    
    @staticmethod
    def compare(current, best):
        return current > best

class DEO(AbstractMeter):
    def __init__(self, args, meters_dict):
        pass

    def compute(self, predicted, target, group):
        pass

    @staticmethod
    def compare(current, best):
        return current < best

class DEOdds(AbstractMeter):
    def __init__(self, args, meters_dict):
        pass

    def compute(self, predicted, target, group):
        pass

    @staticmethod
    def compare(current, best):
        return current < best