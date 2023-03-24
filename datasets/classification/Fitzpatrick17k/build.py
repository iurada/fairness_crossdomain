import os
import random
import pandas as pd

def build_splits(args):
    # args.data_path shall point to data/fitzpatrick17k
    # File structure:
    # data/fitzpatrick17k/
    # |-- images/
    #     |-- img1.jpg
    #     |-- img2.jpg
    #     |-- ...
    # |-- fitzpatrick17k.csv

    df = pd.read_csv(os.path.join(args.data_path, 'fitzpatrick17k.csv'))

    skip = [
        'http://atlasdermatologico.com.br/img?imageId=4030',
        'http://atlasdermatologico.com.br/img?imageId=8363',
        'http://atlasdermatologico.com.br/img?imageId=8362',
        'http://atlasdermatologico.com.br/img?imageId=4505',
        'http://atlasdermatologico.com.br/img?imageId=4031',
        'http://atlasdermatologico.com.br/img?imageId=6724',
        'http://atlasdermatologico.com.br/img?imageId=2767',
        'http://atlasdermatologico.com.br/img?imageId=2766',
        'http://atlasdermatologico.com.br/img?imageId=8364'
    ]

    examples = []
    for index, row in df.iterrows():
        try: 
            if row['url'].strip() in skip: continue # Filter images without url
        except AttributeError: continue

        if int(row['fitzpatrick_scale']) not in [1, 5, 6]: continue # Filter images without fitzpatrick scale
        
        img_name = os.path.join(args.data_path, 'images', row['url'].strip().split('/')[-1])
        target_attrib = 1 if row['three_partition_label'].strip() == 'malignant' else 0 # (non-neopl. + bening)=0 vs malignant=1
        sensitive_attrib = 1 if int(row['fitzpatrick_scale']) == 1 else 0 # type_1=1 vs (type_5 + type_6)=0

        examples.append([img_name, target_attrib, sensitive_attrib])

    # 80/10/10 split
    N80 = len(examples) * 8 // 10
    N10 = len(examples) * 1 // 10

    random.shuffle(examples)

    train_examples = examples[:N80]
    val_examples = examples[N80:N80+N10]
    test_examples = examples[N80+N10:]

    train_imgs = []
    train_labels = []
    for data in train_examples:
        img, ta, sa = data
        train_imgs.append(img)
        train_labels.append([ta, sa])

    val_imgs = []
    val_labels = []
    for data in val_examples:
        img, ta, sa = data
        val_imgs.append(img)
        val_labels.append([ta, sa])

    test_imgs = []
    test_labels = []
    for data in test_examples:
        img, ta, sa = data
        test_imgs.append(img)
        test_labels.append([ta, sa])

    train_set = []
    val_set = []
    test_set = []

    for list_imgs, list_labels, list_set in zip(
        [train_imgs, val_imgs, test_imgs],
        [train_labels, val_labels, test_labels],
        [train_set, val_set, test_set]):

        for img_path, label_group in zip(list_imgs, list_labels):
            label, group = label_group
            list_set.append([img_path, label, group])

    return {'train_set': train_set, 'val_set': val_set, 'test_set': test_set}
