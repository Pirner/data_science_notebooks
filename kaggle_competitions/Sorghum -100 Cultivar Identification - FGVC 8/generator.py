import os

import tensorflow as tf
import pandas as pd
import cv2
import numpy as np


class FGVC8Generator(tf.keras.utils.Sequence):
    def __init__(self,
                 df: pd.DataFrame,
                 im_src: str,
                 batch_size: int,
                 n_classes: int,
                 class_names,
                 input_size=(224, 224, 3),
                 shuffle=True
                 ):
        """
        custom data generator to consume cloud classification data
        :param df: dataframe which contains the labels
        :param im_src: directory where the images sit
        :param class_names: name mappings of the classes
        :param batch_size: number of image to consume in a single batch
        :param input_size: resizing factor the images
        :param shuffle: shuffle the data after every epoch
        """
        self.df = df.copy()
        self.im_src = im_src
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.input_size = input_size
        self.shuffle = shuffle
        self.class_names = class_names

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
        batches = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        ims = []
        labels = []

        for _, row in batches.iterrows():
            tmp = row
            im_path = os.path.join(self.im_src, tmp['image'])
            im = cv2.imread(im_path)
            im = cv2.resize(im, (self.input_size[0], self.input_size[1]))
            entry_labels = tmp['labels']
            label = np.zeros(self.n_classes, dtype=np.float32)
            for key in self.class_names:
                if self.class_names[key].lower() in entry_labels.lower():
                    label[key] = 1.

            ims.append(im)
            labels.append(label)

        x = np.array(ims)
        y = np.array(labels)

        return x, y

    def __len__(self):
        return self.n // self.batch_size
