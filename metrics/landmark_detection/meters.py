import torch
from metrics.utils import AbstractMeter

# Remember to sort meters according to the computational order 
# -- This will be the full presentation order:
METRICS_ORDER = ['SDR', 'MGS', 'mGS', 'DS', 'DTO', 'DeltaDTO', 'HF']

class SDR(AbstractMeter):
    pass

class MGS(AbstractMeter):
    pass

class mGS(AbstractMeter):
    pass

class DS(AbstractMeter):
    pass

class DTO(AbstractMeter):
    pass

class DeltaDTO(AbstractMeter):
    pass

class HF(AbstractMeter):
    pass
