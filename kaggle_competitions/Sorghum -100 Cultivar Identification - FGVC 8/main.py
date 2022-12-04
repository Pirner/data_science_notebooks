# competition link https://www.kaggle.com/competitions/sorghum-id-fgvc-9/overview
import os

import pandas as pd

from generator import FGVC8Generator
from training import Trainer


def main():
    root_path = r'C:\kaggle\plant-pathology-2021-fgvc8'
    df = pd.read_csv(os.path.join(root_path, 'train.csv'))
    labels = df['labels'].unique()
    class_names = {
        0: 'complex',
        1: 'scab',
        2: 'frog_eye_leaf_spot',
        3: 'rust',
        4: 'powdery_mildew',
        6: 'healthy'
    }

    n_classes = 7

    train_gen = FGVC8Generator(
        df=df,
        im_src=os.path.join(root_path, 'train_images'),
        batch_size=16,
        class_names=class_names,
        n_classes=n_classes,
    )

    trainer = Trainer()
    trainer.train_model(train_gen=train_gen, classes=n_classes)

    print('Goodbye World')


if __name__ == '__main__':
    main()
