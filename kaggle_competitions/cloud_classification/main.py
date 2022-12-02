import os

import pandas as pd

from kaggle_competitions.cloud_classification.generator import CloudDataGen


def main():
    data_path = r'C:\kaggle\cloud_classification\cloud-type-classification-3'
    data_file = os.path.join(data_path, 'train.csv')
    df = pd.read_csv(data_file)

    # 7 classes in total, 0 -> 6 (inclusive)
    classes = df['label'].unique()
    # create train gen
    train_gen = CloudDataGen(
        df=df,
        im_src=os.path.join(data_path, 'images', 'train'),
        batch_size=2,
    )

    for x, y in train_gen:
        exit(0)

    print('Goodbye World')


if __name__ == '__main__':
    main()
