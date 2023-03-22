import os
from torch.utils.data import DataLoader

def build_dataloaders(args):
    # Load dataset build module
    module_name = '.'.join(os.path.normpath(args.dataset).split(os.sep))
    dataset_module = __import__(f'{module_name}.build', fromlist=[module_name])

    # Load experiment module
    module_name = '.'.join(os.path.normpath(args.experiment).split(os.sep))
    experiment_module = __import__(f'{module_name}.experiment', fromlist=[module_name])

    # Load base_datasets & base_transforms module
    module_name = f'datasets.{args.experiment_type}'

    splits_dict = dataset_module.build_splits(args)
    data_config = experiment_module.Experiment.data_config

    loaders = {}

    for loader_name, loader_config in data_config.items():

        # Load set of examples
        examples_set = splits_dict[loader_config['set']]

        # Optionally filter examples
        if loader_config['filter'] is not None:
            examples_set = loader_config['filter'](examples_set)

        # Load Transform object
        exec(f'from {module_name}.base_transforms import {loader_config["transform"]}')
        transform = eval(f'{loader_config["transform"]}().build_transform(args)')

        # Load Dataset object
        exec(f'from {module_name}.base_datasets import {loader_config["dataset"]}')
        dataset = eval(f'{loader_config["dataset"]}(examples_set, transform, args)')

        loaders[loader_name] = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                          pin_memory=args.pin_memory, shuffle=loader_config['shuffle'], 
                                          drop_last=loader_config['drop_last'])

    return loaders
