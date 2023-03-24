import torch
from sklearn.metrics import confusion_matrix

# Remember to sort meters according to the computational order 
# -- This will be the full presentation order:
METRICS_ORDER = ['Acc', 'MGA', 'mGA', 'DA', 'DEO', 'DEOdds', 'DTO', 'DeltaDTO', 'HF']

class Acc:
    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None

    def compute(self, predicted, target, group):
        predicted = torch.argmax(predicted, dim=-1)
        self.value = (predicted == target).sum().item() / predicted.size(0)

        acc_gr0 = (predicted[group == 0] == target[group == 0]).sum().item() / predicted[group == 0].size(0)
        acc_gr1 = (predicted[group == 1] == target[group == 1]).sum().item() / predicted[group == 1].size(0)

        if 'MGA' in self.meters_dict.keys():
            self.meters_dict['MGA'].value = max(acc_gr0, acc_gr1)
        if 'mGA' in self.meters_dict.keys():
            self.meters_dict['mGA'].value = min(acc_gr0, acc_gr1)

        return self.value

    @staticmethod
    def compare(current, best):
        return current > best

class MGA:
    required_metrics = ['Acc']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
    
    def compute(self, predicted, target, group):
        return self.value
    
    @staticmethod
    def compare(current, best):
        return current > best
    
class mGA:
    required_metrics = ['Acc']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None

    def compute(self, predicted, target, group):
        return self.value

    @staticmethod
    def compare(current, best):
        return current > best

class DA:
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

class HF:
    required_metrics = ['MGA', 'DA']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
        self.baseline_MGA = args.baseline_MGA
        self.baseline_DA = args.baseline_DA

    def compute(self, predicted, target, group):
        MGA = self.meters_dict['MGA'].value * 100
        DA = self.meters_dict['DA'].value * 100
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

class DTO:
    required_metrics = ['MGA', 'mGA']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None

    def compute(self, predicted, target, group):
        MGA = self.meters_dict['MGA'].value * 100
        mGA = self.meters_dict['mGA'].value * 100
        self.value = ((100 - MGA)**2 + (100 - mGA)**2)**0.5
        return self.value

    @staticmethod
    def compare(current, best):
        return current < best

class DeltaDTO:
    required_metrics = ['MGA', 'mGA']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
        self.baseline_DTO = args.baseline_DTO

    def compute(self, predicted, target, group):
        MGA = self.meters_dict['MGA'].value * 100
        mGA = self.meters_dict['mGA'].value * 100
        self.value = self.baseline_DTO - ((100 - MGA)**2 + (100 - mGA)**2)**0.5
        return self.value

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--baseline_DTO', type=float, required=True)
    
    @staticmethod
    def compare(current, best):
        return current > best

class DEO:

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
        self.deodds = None

    def compute(self, predicted, target, group):
        predicted = torch.argmax(predicted, dim=-1)
        target_gr0 = target[group == 0]
        target_gr1 = target[group == 1]
        predicted_gr0 = predicted[group == 0]
        predicted_gr1 = predicted[group == 1]

        tn_0, fp_0, fn_0, tp_0 = confusion_matrix(
            target_gr0.numpy(), predicted_gr0.numpy()).ravel()
        
        tn_1, fp_1, fn_1, tp_1 = confusion_matrix(
            target_gr1.numpy(), predicted_gr1.numpy()).ravel()

        TPR_prot_attr_0 = tp_0 / (tp_0 + fn_0)
        TPR_prot_attr_1 = tp_1 / (tp_1 + fn_1)
        FPR_prot_attr_0 = fp_0 / (fp_0 + tn_0)
        FPR_prot_attr_1 = fp_1 / (fp_1 + tn_1)

        self.value = abs(TPR_prot_attr_0 - TPR_prot_attr_1)
        self.deodds = self.value + abs(FPR_prot_attr_0 - FPR_prot_attr_1)
        return self.value

    @staticmethod
    def compare(current, best):
        return current < best

class DEOdds:
    required_metrics = ['DEO']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None

    def compute(self, predicted, target, group):
        self.value = self.meters_dict['DEO'].deodds
        return self.value

    @staticmethod
    def compare(current, best):
        return current < best