import os
import random

def additional_arguments(parser):
    parser.add_argument('--target_attribute', type=int, choices=list(range(40)))
    parser.add_argument('--protected_attribute', type=int, choices=list(range(40)))
    parser.add_argument('--transfer_experiment', type=str, choices=['age2gender', 'gender2age', 'gender2gender', 'age2age'])
    return ['target_attribute', 'protected_attribute', 'transfer_experiment']

def build_splits(args):
    # args.data_path shall point to data/celeba
        # File structure:
        # data/celeba/
        # |-- img_align_celeba/
        #     |-- img1.jpg
        #     |-- img2.jpg
        #     |-- ...
        # |-- list_attr_celeba.txt
        # |-- list_eval_partition.txt

        if args.transfer_experiment is None:
            assert args.target_attribute is not None, '[CelebA] --target_attribute not set.'
            assert args.protected_attribute is not None, '[CelebA] --protected_attribute not set.'

        val_begin_idx = 162770
        test_begin_idx = 182637

        AGE = 39+1
        GENDER = 20+1

        protected_attribute_train = args.protected_attribute
        protected_attribute_val = args.protected_attribute
        protected_attribute_test = args.protected_attribute

        if args.transfer_experiment is not None:
            if args.transfer_experiment == 'age2gender':
                protected_attribute_train = AGE
                protected_attribute_val = AGE
                protected_attribute_test = GENDER
            elif args.transfer_experiment == 'gender2age':
                protected_attribute_train = GENDER
                protected_attribute_val = GENDER
                protected_attribute_test = AGE
            elif args.transfer_experiment == 'gender2gender':
                protected_attribute_train = GENDER
                protected_attribute_val = GENDER
                protected_attribute_test = GENDER
            elif args.transfer_experiment == 'age2age':
                protected_attribute_train = AGE
                protected_attribute_val = AGE
                protected_attribute_test = AGE

        with open(os.path.join(args.data_path, 'list_attr_celeba.txt'), 'r') as f:
            lines = f.readlines()[2:]

        train_set = []
        val_set = []
        test_set = []

        for i, line in enumerate(lines):
            tmp = line.split()
            file_name = tmp[0]

            img_path = os.path.join(args.data_path, 'img_align_celeba', file_name)
            target_attribute_value = max(0, int(tmp[args.target_attribute + 1]))
            
            if i < val_begin_idx:
                protected_attribute_value = max(0, int(tmp[protected_attribute_train + 1]))
                train_set.append([img_path, target_attribute_value, protected_attribute_value])
            elif i < test_begin_idx:
                protected_attribute_value = max(0, int(tmp[protected_attribute_val + 1]))
                val_set.append([img_path, target_attribute_value, protected_attribute_value])
            else:
                protected_attribute_value = max(0, int(tmp[protected_attribute_test + 1]))
                test_set.append([img_path, target_attribute_value, protected_attribute_value])
        
        train_set = random.sample(train_set, 10000)

        return {'train_set': train_set, 'val_set': val_set, 'test_set': test_set}
