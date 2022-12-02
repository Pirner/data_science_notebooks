import os

import tensorflow as tf
import pandas as pd
import cv2


class CloudDataGen(tf.keras.utils.Sequence):
    def __init__(self,
                 df: pd.DataFrame,
                 im_src: str,
                 batch_size: int,
                 input_size=(224, 224, 3),
                 shuffle=True
                 ):
        """
        custom data generator to consume cloud classification data
        :param df: dataframe which contains the labels
        :param im_src: directory where the images sit
        :param batch_size: number of image to consume in a single batch
        :param input_size: resizing factor the images
        :param shuffle: shuffle the data after every epoch
        """
        self.df = df.copy()
        self.im_src = im_src
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.df)

        # self._init_runtime_loading()

    def _init_runtime_loading(self):
        """
        initializes the data to consume images with on training resizing
        :return:
        """
        tmp = self.df.iloc[0]
        im_path = os.path.join(self.im_src, tmp['id'])
        im = cv2.imread(im_path)
        exit(0)

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        exit(0)

    def __len__(self):
        return self.n // self.batch_size
