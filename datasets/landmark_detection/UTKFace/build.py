import os
import random
import numpy as np

def additional_arguments(parser):
    parser.add_argument('--experiment', type=str, choices=['age', 'skintone', 'age2skin', 'skin2age', 'skin2skin', 'age2age'], default='skintone')
    return ['experiment']

def build_splits(args):
    # args.data_path shall point to data/utkface
    # File structure:
    # data/utkface/
    # |-- UTKFace/
    #     |-- img1.jpg
    #     |-- img2.jpg
    #     |-- ...
    # |-- utk_train.txt
    # |-- utk_val.txt
    # |-- ...
    
    if args.experiment == 'skintone':
        train_file = 'utk_train.txt'
        val_file = 'utk_val.txt'
        test_file = 'utk_test.txt'
    elif args.experiment == 'age':
        train_file = 'utk_age_train.txt'
        val_file = 'utk_age_val.txt'
        test_file = 'utk_age_test.txt'
    else:
        train_file = 'utk_transfer_train.txt'
        val_file = 'utk_transfer_val.txt'
        test_file = 'utk_transfer_test.txt'
    
    train_set = []
    val_set = []
    test_set = []

    with open(os.path.join(args.data_path, train_file), 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split()
        name = line[0]
        
        if args.experiment in ['skintone', 'skin2age', 'skin2skin']:
            group = int(name.split('_')[2])
        elif args.experiment == ['age', 'age2skin', 'age2age']:
            group = 0 if int(name.split('_')[0]) <= 10 else 1

        name = os.path.join(args.data_path, 'UTKFace', name)
        landmarks = np.array([int(k) for k in line[1:]])
        train_set.append([name, landmarks, group])

    with open(os.path.join(args.data_path, val_file), 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split()
        name = line[0]
        
        if args.experiment in ['skintone', 'skin2age', 'skin2skin']:
            group = int(name.split('_')[2])
        elif args.experiment == ['age', 'age2skin', 'age2age']:
            group = 0 if int(name.split('_')[0]) <= 10 else 1

        name = os.path.join(args.data_path, 'UTKFace', name)
        landmarks = np.array([int(k) for k in line[1:]])
        val_set.append([name, landmarks, group])

    with open(os.path.join(args.data_path, test_file), 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip().split()
        name = line[0]
        
        if args.experiment in ['skintone', 'age2skin', 'skin2skin']:
            group = int(name.split('_')[2])
        elif args.experiment == ['age', 'skin2age', 'age2age']:
            group = 0 if int(name.split('_')[0]) <= 10 else 1

        name = os.path.join(args.data_path, 'UTKFace', name)
        landmarks = np.array([int(k) for k in line[1:]])
        test_set.append([name, landmarks, group])

    return {'train_set': train_set, 'val_set': val_set, 'test_set': test_set}
