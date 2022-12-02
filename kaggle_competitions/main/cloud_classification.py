import os

import pandas as pd


def main():
    data_path = r'C:\kaggle\cloud_classification\cloud-type-classification-3'
    data_file = os.path.join(data_path, 'train.csv')
    df = pd.read_csv(data_file)
    print('Goodbye World')


if __name__ == '__main__':
    main()
