import os
from argparse import ArgumentParser
import metrics.classification.meters as classification_meters
import metrics.landmark_detection.meters as landmark_detection_meters

def parse_arguments():
    parser = ArgumentParser()

    #! General arguments
    parser.add_argument('--experiment', type=str, help='Path to the experiment folder to run.', required=True)
    parser.add_argument('--dataset', type=str, help='Path to the dataset folder to use.', required=True)

    parser.add_argument('--max_iters', type=int, help='Total number of training iterations.', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size passed to DataLoaders.', required=True)
    parser.add_argument('--num_workers', type=int, help='How many workers to spawn with DataLoaders.', required=True)
    parser.add_argument('--pin_memory', action='store_true', help='DataLoader argument.')

    parser.add_argument('--validate_every', type=int, help='How frequently (in terms of number of iterations) to run validation.', required=True)
    parser.add_argument('--tracked_metrics', nargs='+', help='Which metrics to track. (See `metrics/` folder for the full list of collectible metrics)', required=True)
    parser.add_argument('--model_selection', type=str, help='Which metric to use for model selection. (See `metrics/` folder for the full list of collectible metrics)', required=True)

    parser.add_argument('--data_path', type=str, help='Folder where data is stored on disk.', required=True)
    parser.add_argument('--log_path', type=str, help='Folder where to save logs and checkpoints.', default='')

    parser.add_argument('--cpu', action='store_true')

    args, _ = parser.parse_known_args()

    #! Compatibility checks
    exp_type = os.path.normpath(args.experiment).split(os.sep)[1]
    dset_type = os.path.normpath(args.dataset).split(os.sep)[1]
    assert exp_type == dset_type, f'Experiment type is "{exp_type}" but Dataset type is "{dset_type}".'

    collectible_metrics_classification = [k for k in classification_meters.__dict__ if not k.startswith('__')]
    collectible_metrics_landmark_detection = [k for k in landmark_detection_meters.__dict__ if not k.startswith('__')]
    ref_metrics = eval(f'collectible_metrics_{exp_type}')
    for metric in args.tracked_metrics:
        if metric not in ref_metrics:
            raise ValueError(f'"{metric}" metric cannot be collected for "{exp_type}"')
    
    assert args.model_selection in args.tracked_metrics, 'Model selection metric must be a tracked metric'

    if args.model_selection not in ref_metrics:
        raise ValueError(f'"{args.model_selection}" metric cannot be used for model selection in "{exp_type}"')
    
    #! Solve metrics dependencies
    current_tracked_metrics = args.tracked_metrics.copy()

    def add_required_metrics(required_metrics, tracked_metrics):
        for metric in required_metrics:
            if metric not in tracked_metrics:
                tracked_metrics.append(metric)
                obj = eval(f'{exp_type}_meters.{metric}')
                if hasattr(obj, 'required_metrics'):
                    add_required_metrics(obj.required_metrics, tracked_metrics)

    for metric in current_tracked_metrics:
        obj = eval(f'{exp_type}_meters.{metric}')
        if hasattr(obj, 'required_metrics'):
            add_required_metrics(obj.required_metrics, current_tracked_metrics)

    #! [Metrics] Additional arguments
    for metric in current_tracked_metrics:
        obj = eval(f'{exp_type}_meters.{metric}')

        if hasattr(obj, 'additional_arguments'):
            obj.additional_arguments(parser)

    #! [Datasets] Additional arguments
    dset_name = os.path.normpath(args.dataset).split(os.sep)[2]
    module_name = '.'.join(os.path.normpath(args.dataset).split(os.sep))
    dataset_module = __import__(f'{module_name}.dataset', fromlist=[module_name])
    obj = dataset_module
    if hasattr(obj, 'additional_arguments'):
        obj.additional_arguments(parser)

    #! [Experiments] Additional arguments
    exp_name = os.path.normpath(args.experiment).split(os.sep)[2]
    module_name = '.'.join(os.path.normpath(args.experiment).split(os.sep))
    experiment_module = __import__(f'{module_name}.experiment', fromlist=[module_name])
    obj = experiment_module.Experiment
    if hasattr(obj, 'additional_arguments'):
        obj.additional_arguments(parser)

    args = parser.parse_args()

    #! Set additional attributes of the current Namespace
    args.experiment_type = exp_type
    args.tracked_metrics = current_tracked_metrics.copy()

    # Set output path + name 
    #TODO

    #print(args)
    #exit()

    return args