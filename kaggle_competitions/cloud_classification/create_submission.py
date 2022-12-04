import os
from tqdm import tqdm

import tensorflow as tf
import pandas as pd
import cv2
import numpy as np


def main():
    data_path = r'C:\kaggle\cloud_classification\cloud-type-classification-3'
    model_path = r'resnet_model.h5'
    df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))

    model = tf.keras.models.load_model(model_path)

    for i in tqdm(range(len(df_test)), total=len(df_test)):
        row = df_test.iloc[i]
        im_path = os.path.join(data_path, 'images', 'test', row['id'])
        im = cv2.imread(im_path)

        im = cv2.resize(im, (224, 224))
        # im = im.astype(np.float32) / 255.
        # im = np.array(im)
        im = im.astype(np.float32)
        im = np.expand_dims(im, axis=0)
        # predict = model(im)
        # predict = int(np.argmax(predict))
        df_test.loc[i, 'predict'] = 2

    df_test.to_csv('subm.csv', index=False)


if __name__ == '__main__':
    main()
