class AbstractMeter:
    def __init__(self, args, meters_dict):
        raise NotImplementedError()
    
    def compute(self, predicted, target, group):
        raise NotImplementedError()
    
def build_meters_dict(args):
    meter_module = __import__(f'metrics.{args.experiment_type}.meters', fromlist=[f'metrics.{args.experiment_type}'])
    order = meter_module.METRICS_ORDER
    meters_dict = {k: eval(f'{meter_module}.{k}({args})') for k in order if k in args.tracked_metrics}
    return meters_dict

def collect_metrics(meters_dict, predicted, target, group):
    output = {}
    predicted = predicted.cpu()
    target = target.cpu()
    group = group.cpu()
    for name, meter in meters_dict.items():
        output[name] = meter.compute(predicted, target, group)
    return output