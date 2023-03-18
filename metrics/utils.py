
def build_meters_dict(args):
    exec(f'from metrics.{args.experiment_type} import meters')
    order = eval(f'meters.METRICS_ORDER')

    meters_dict = {}
    for k in order:
        if k in args.tracked_metrics:
            meters_dict[k] = eval(f'meters.{k}(args, meters_dict)')

    return meters_dict

def collect_metrics(meters_dict, predicted, target, group):
    output = {}
    predicted = predicted.cpu()
    target = target.cpu()
    group = group.cpu()
    for name, meter in meters_dict.items():
        output[name] = meter.compute(predicted, target, group)
    return output