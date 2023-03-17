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
    base_datasets_module = __import__(f'{module_name}.base_datasets', fromlist=[module_name])
    base_transforms_module = __import__(f'{module_name}.base_transforms', fromlist=[module_name])

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
        transform = eval(f'datasets.{args.experiment_type}.base_transforms.{loader_config["transform"]}()')

        # Load Dataset object
        dataset = eval(f'datasets.{args.experiment_type}.base_datasets.{loader_config["dataset"]}(examples_set, transform)')

        loaders[loader_name] = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                          pin_memory=args.pin_memory, shuffle=loader_config['shuffle'], 
                                          drop_last=loader_config['drop_last'])

    return loaders
