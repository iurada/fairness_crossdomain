import torch
import numpy as np

# Remember to sort meters according to the computational order 
# -- This will be the full presentation order:
METRICS_ORDER = ['SDR', 'MGS', 'mGS', 'DS', 'DTO', 'DeltaDTO', 'HF']

def compute_nme(lms_pred, lms_gt, norm_idx1=36, norm_idx2=45):
    norm = np.linalg.norm(lms_gt.reshape(-1, 2)[norm_idx1] - lms_gt.reshape(-1, 2)[norm_idx2]) 
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))
    nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm 
    return nme

def compute_sdr(nmes, thres=0.08, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    sdr = ys[-1]
    return sdr

class SDR:

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
        self.thresh = args.SDR_threshold
        self.p1_index = args.p1_NME_index
        self.p2_index = args.p2_NME_index

    def compute(self, predicted, target, group):
        predicted = predicted.numpy()
        target = target.numpy()
        group = group.numpy()

        nmes_overall = []
        nmes_gr0 = []
        nmes_gr1 = []
        for pred, targ, g in zip(predicted, target, group):
            nme = compute_nme(pred, targ, self.p1_index, self.p2_index)
            nmes_overall.append(nme)
            if g == 0:
                nmes_gr0.append(nme)
            else:
                nmes_gr1.append(nme)
            
        nmes_overall = np.stack(nmes_overall, axis=0)
        nmes_gr0 = np.stack(nmes_gr0, axis=0)
        nmes_gr1 = np.stack(nmes_gr1, axis=0)

        self.value = compute_sdr(nmes_overall, thres=self.thresh)
        sdr_gr0 = compute_sdr(nmes_gr0, thres=self.thresh)
        sdr_gr1 = compute_sdr(nmes_gr1, thres=self.thresh)

        if 'MGS' in self.meters_dict.keys():
            self.meters_dict['MGS'].value = max(sdr_gr0, sdr_gr1)
        if 'mGS' in self.meters_dict.keys():
            self.meters_dict['mGS'].value = min(sdr_gr0, sdr_gr1)

        return self.value

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--SDR_threshold', type=float, required=True, help='Threshold [0.0, 1.0] to be used for computing SDR.')
        parser.add_argument('--p1_NME_index', type=int, required=True, help='Index of first point to compute normalization factor for NME.')
        parser.add_argument('--p2_NME_index', type=int, required=True, help='Index of second point to compute normalization factor for NME.')
    
    @staticmethod
    def compare(current, best):
        return current > best

class MGS:
    required_metrics = ['SDR']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
    
    def compute(self, predicted, target, group):
        return self.value
    
    @staticmethod
    def compare(current, best):
        return current > best

class mGS:
    required_metrics = ['SDR']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
    
    def compute(self, predicted, target, group):
        return self.value
    
    @staticmethod
    def compare(current, best):
        return current > best

class DS:
    required_metrics = ['MGS', 'mGS']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None

    def compute(self, predicted, target, group):
        self.value = self.meters_dict['MGS'].value - self.meters_dict['mGS'].value
        return self.value
    
    @staticmethod
    def compare(current, best):
        return current < best

class DTO:
    required_metrics = ['MGS', 'mGS']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None

    def compute(self, predicted, target, group):
        MGS = self.meters_dict['MGS'].value * 100
        mGS = self.meters_dict['mGS'].value * 100
        self.value = ((100 - MGS)**2 + (100 - mGS)**2)**0.5
        return self.value

    @staticmethod
    def compare(current, best):
        return current < best

class DeltaDTO:
    required_metrics = ['MGS', 'mGS']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
        self.baseline_DTO = args.baseline_DTO

    def compute(self, predicted, target, group):
        MGS = self.meters_dict['MGS'].value * 100
        mGS = self.meters_dict['mGS'].value * 100
        self.value = self.baseline_DTO - ((100 - MGS)**2 + (100 - mGS)**2)**0.5
        return self.value

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--baseline_DTO', type=float, required=True)
    
    @staticmethod
    def compare(current, best):
        return current > best

class HF:
    required_metrics = ['MGS', 'DS']

    def __init__(self, args, meters_dict):
        self.meters_dict = meters_dict
        self.value = None
        self.baseline_MGS = args.baseline_MGS
        self.baseline_DS = args.baseline_DS

    def compute(self, predicted, target, group):
        MGS = self.meters_dict['MGS'].value * 100
        DS = self.meters_dict['DS'].value * 100
        a = (100 + MGS - self.baseline_MGS) / 2
        b = (100 + self.baseline_DS - DS) / 2
        self.value = (2 * a * b) / (a + b)
        return self.value

    @staticmethod
    def additional_arguments(parser):
        parser.add_argument('--baseline_MGS', type=float, required=True)
        parser.add_argument('--baseline_DS', type=float, required=True)
    
    @staticmethod
    def compare(current, best):
        return current > best
