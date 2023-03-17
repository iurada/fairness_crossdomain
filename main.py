import os
import torch
import logging
from parse_args import parse_arguments
from metrics.utils import build_meters_dict, collect_metrics
from datasets.utils import build_dataloaders

DEVICE = None

def load_experiment(args, dataloaders):
    module_name = '.'.join(os.path.normpath(args.experiment).split(os.sep))
    experiment_module = __import__(f'{module_name}.experiment', fromlist=[module_name])
    return experiment_module.Experiment(args, dataloaders)

def main():
    global DEVICE

    args = parse_arguments()

    # Setup logger
    logging.basicConfig(filename=os.path.join(args.log_path, 'log.txt'), format='%(message)s', level=logging.INFO, filemode='a')

    DEVICE = torch.device('cpu')
    if torch.cuda.is_available() and not args.cpu:
        DEVICE = torch.device('cuda:0')

    # Data setup
    dataloaders = build_dataloaders(args)
    
    # Experiments setup
    experiment = load_experiment(args, dataloaders)

    # Optionally Resume checkpoint
    if os.path.exists(args.log_path):
        experiment.load(os.path.join(args.log_path, 'current.pth'))
    
    # Meters setup
    meters_dict = build_meters_dict(args)

    # Training Loop
    while experiment.iteration < args.max_iter:

        for data in dataloaders['train']:

            train_losses = experiment.train_iteration(data)

            # Log losses eg. via wandb
            logging.info(train_losses)

            # Validation phase
            if experiment.iteration % args.validate_every:
                predicted, target, group = experiment.evaluate(dataloaders['val'])
                metrics = collect_metrics(meters_dict, predicted, target, group)

                # Log metrics eg. via wandb
                logging.info(f'[VAL @ {experiment.iteration}] {metrics}')

                if experiment.best_metric is None or meters_dict[args.model_selection].compare(metrics[args.model_selection], experiment.best_metric):
                    experiment.best_metric = metrics[args.model_selection]
                    experiment.save(os.path.join(args.log_path, 'best.pth'))
                experiment.save(os.path.join(args.log_path, 'current.pth'))
                
            experiment.iteration += 1
            if experiment.iteration >= args.max_iter: break
    
    # Test phase
    experiment.load(os.path.join(args.log_path, 'best.pth'))
    predicted, target, group = experiment.evaluate(dataloaders['test'])
    metrics = collect_metrics(meters_dict, predicted, target, group)

    # Log metrics eg. via wandb
    logging.info(f'[TEST] {metrics}')

if __name__ == '__main__':
    main()