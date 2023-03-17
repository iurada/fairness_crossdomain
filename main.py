import os
import torch
from parse_args import parse_arguments

from metrics.utils import build_meters_dict, collect_metrics

DEVICE = None

def load_experiment(args):
    module_name = '.'.join(os.path.normpath(args.experiment).split(os.sep))
    experiment_module = __import__(f'{module_name}.experiment', fromlist=[module_name])
    return experiment_module.Experiment(args)

def load_dataset(ctx, dataset):
    module_name = '.'.join(os.path.normpath(dataset).split(os.sep))
    dataset_module = __import__(f'{module_name}.dataset', fromlist=[module_name])
    print(dataset)
    #return dataset_module.build_dataloaders(args)



def main():
    global DEVICE

    args = parse_arguments()

    DEVICE = torch.device('cpu')
    if torch.cuda.is_available() and not args.cpu:
        DEVICE = torch.device('cuda:0')

    experiment = load_experiment(args)

    print(experiment)

    exit()
    dataloaders = load_dataset(args)
    
    
    iteration = 0
    best_metric = None
    train_loss = None

    # Meters setup
    meters_dict = build_meters_dict(args)

    # Training Loop
    while iteration < args.max_iter:

        for data in dataloaders['train']:

            if train_loss is None:
                train_loss = experiment.train_iteration(data)
            else:
                train_loss += experiment.train_iteration(data)

            if iteration % args.print_every:
                pass

            if iteration % args.validate_every:
                predicted, target, group = experiment.evaluate(dataloaders['val'])
                metrics = collect_metrics(meters_dict, predicted, target, group)

                if meters_dict[args.tracked_metric].compare(metrics[args.tracked_metric], experiment.best_metric):
                    experiment.best_metric = metrics[args.tracked_metric]
                

            iteration += 1
            if iteration >= args.max_iter: break


    

    

    

if __name__ == '__main__':
    main()