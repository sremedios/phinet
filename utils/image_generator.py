'''
Data generator which works with nifti files

TODO: patch collection from images
'''
from keras.utils import Sequence, to_categorical
import numpy as np
import nibabel as nib
import os

class DataGenerator(Sequence):
    def __init__(self, list_IDs, batch_size, dim, n_channels,
                 n_classes, class_encodings, shuffle):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.class_encodings = class_encodings
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes at start of training and after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty(
            (self.batch_size, *self.dim, self.n_channels))
        y = np.empty(
            (self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img = nib.load(ID).get_data()
            X[i,] = np.reshape(img, img.shape + (1,))

            y[i] = to_categorical(self.class_encodings[ID.split(os.sep)[-2]], 
                                  num_classes=self.n_classes) 


        return X, y
