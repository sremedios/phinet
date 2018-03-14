'''
Samuel Remedios
NIH CC CNRM
Train PhiNet to classify MRI as T1, T2, FLAIR
'''

import os
import numpy as np
import shutil
from sklearn.utils import shuffle
from datetime import datetime
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.models import load_model
from models.phinet import phinet
from utils.preprocessing import load_data

def now():
    '''
    Returns a string format of current time, for use in
    checkpoint file naming
    '''
    return datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")

############### DIRECTORIES ###############

TRAIN_DIR = os.path.join("data", "train")
PREPROCESSED_TMP_DIR = os.path.join("data", "robustfov")
WEIGHT_DIR = "weights"

############### DATA IMPORT ###############

PATCH_SIZE = (45,45,15)
X, y, _ = load_data(TRAIN_DIR, PREPROCESSED_TMP_DIR, PATCH_SIZE)
X, y = shuffle(X, y, random_state=0)

num_classes = len(y[0])
print("Finished data processing")

############### MODEL SELECTION ###############

LR = 1e-4
LOAD_WEIGHTS = False

if not os.path.exists(WEIGHT_DIR):
    os.makedirs(WEIGHT_DIR)

if LOAD_WEIGHTS:
    weight_files = os.listdir(WEIGHT_DIR)
    weight_files.sort()
    best_weights = os.path.join(WEIGHT_DIR, weight_files[-1])
    model = load_model(best_weights)
else:
    model = phinet(input_shape=PATCH_SIZE, n_classes=num_classes, learning_rate=LR)

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

# Early Stopping, used to quantify convergence
# convergence is defined as no improvement by 1e-4 for 10 consecutive epochs
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
callbacks_list.append(es)

############### TRAINING ###############
# the number of epochs is set high so that EarlyStopping can be the terminator
NB_EPOCHS = 10000000
BATCH_SIZE = 128 

model.fit(X, y, epochs=NB_EPOCHS, validation_split=0.2,
          batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)

shutil.rmtree(PREPROCESSED_TMP_DIR)
K.clear_session()
