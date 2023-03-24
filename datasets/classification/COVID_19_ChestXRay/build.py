import os
import pandas as pd

def build_splits(args):
    # args.data_path shall point to data/covid-chestxray-dataset
    # File structure:
    # data/covid-chestxray-dataset/
    # |-- images/
    #     |-- img1.jpg
    #     |-- img2.jpg
    #     |-- ...
    # |-- metadata.csv

    df = pd.read_csv(os.path.join(args.data_path, 'metadata.csv'))
    view_filter = ['AP', 'AP Erect', 'PA', 'AP Supine']

    dset = df[(df['view'] == view_filter[0]) |
            (df['view'] == view_filter[1]) |
            (df['view'] == view_filter[2]) |
            (df['view'] == view_filter[3])]
    
    male_covid = dset[(dset['finding'] == 'Pneumonia/Viral/COVID-19') & (dset['sex'] == 'M')]['filename'].values.tolist()
    female_covid = dset[(dset['finding'] == 'Pneumonia/Viral/COVID-19') & (dset['sex'] == 'F')]['filename'].values.tolist()
    male_noncovid = dset[(dset['finding'] != 'Pneumonia/Viral/COVID-19') & (dset['sex'] == 'M')]['filename'].values.tolist()
    female_noncovid = dset[(dset['finding'] != 'Pneumonia/Viral/COVID-19') & (dset['sex'] == 'F')]['filename'].values.tolist()

    train_imgs = male_covid[:76] + female_covid[:76] + male_noncovid[:76] + female_noncovid[:76]
    train_labels = [[1, 1] for _ in range(76)] + [[1, 0] for _ in range(76)] + [[0, 1] for _ in range(76)] + [[0, 0] for _ in range(76)]

    val_imgs = male_covid[183:183 + 46] + female_covid[92:92 + 24] + male_noncovid[107:107 + 27] + female_noncovid[76:76 + 19]
    val_labels = [[1, 1] for _ in range(46)] + [[1, 0] for _ in range(24)] + [[0, 1] for _ in range(27)] + [[0, 0] for _ in range(19)]

    test_imgs = male_covid[183 + 46:] + female_covid[92 + 24:] + male_noncovid[107 + 27:] + female_noncovid[76 + 19:]
    test_labels = [[1, 1] for _ in range(58)] + [[1, 0] for _ in range(29)] + [[0, 1] for _ in range(34)] + [[0, 0] for _ in range(24)]

    train_set = []
    val_set = []
    test_set = []

    for list_imgs, list_labels, list_set in zip(
        [train_imgs, val_imgs, test_imgs],
        [train_labels, val_labels, test_labels],
        [train_set, val_set, test_set]):

        for img_name, label_group in zip(list_imgs, list_labels):
            img_path = os.path.join(args.data_path, 'images', img_name)
            label, group = label_group
            list_set.append([img_path, label, group])

    return {'train_set': train_set, 'val_set': val_set, 'test_set': test_set}
