'''
Samuel Remedios
NIH CC CNRM
Train PhiNet to classify MRI as T1, T2, FLAIR
'''

import os
import numpy as np
import shutil
from datetime import datetime
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from models.phinet import phinet
from utils.load_data import load_data_multiclass, preprocess_training_data

############### DIRECTORIES ###############

TRAIN_DIR = os.path.join("data", "train")
PREPROCESSED_TMP_DIR = os.path.join("tmp_processing")
WEIGHT_DIR = "weights"

############### DATA PREPROCESSING ###############

preprocess_training_data(TRAIN_DIR, PREPROCESSED_TMP_DIR)

############### DATA IMPORT ###############

def now():
    '''
    Returns a string format of current time, for use in
    checkpoint file naming
    '''
    return datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")

# import data
MODE = "multiclass"
MODEL_VERSION = "phinet"
MODEL_NAME = MODEL_VERSION + "_" + MODE
X, y = load_data_multiclass(PREPROCESSED_TMP_DIR, model_name=MODEL_NAME)
print(X[0].shape)
INPUT_SHAPE = X[0].shape

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)
print("Finished data processing")

############### MODEL SELECTION ###############

WEIGHT_DIR = os.path.join("weights", MODEL_NAME)
LR = 1e-6

LOAD_WEIGHTS = False

if not os.path.exists(WEIGHT_DIR):
    os.makedirs(WEIGHT_DIR)

if LOAD_WEIGHTS:
    weight_files = os.listdir(WEIGHT_DIR)
    weight_files.sort()
    best_weights = os.path.join(WEIGHT_DIR, weight_files[-1])
    model = phinet(input_shape=INPUT_SHAPE, learning_rate=LR,
                   load_weights=True, weights=best_weights)
else:
    model = phinet(input_shape=INPUT_SHAPE, n_inputs=3, learning_rate=LR)

print(model.summary())

############### CALLBACKS ###############

callbacks_list = []

# Checkpoint
fpath = os.path.join(
    WEIGHT_DIR, now()+"-epoch-{epoch:04d}-val_acc-{val_acc:.4f}.hdf5")
checkpoint = ModelCheckpoint(
    fpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list.append(checkpoint)

# Dynamic Learning Rate
dlr = ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=5,
                        mode='max', verbose=1, cooldown=5, min_lr=1e-8)
callbacks_list.append(dlr)

############### TRAINING ###############

NB_EPOCHS = 10

model.fit(X, y, epochs=NB_EPOCHS, validation_split=0.2,
          batch_size=16, verbose=1, callbacks=callbacks_list)

shutil.rmtree(PREPROCESSED_TMP_DIR)
K.clear_session()
