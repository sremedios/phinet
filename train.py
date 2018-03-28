'''
Samuel Remedios
NIH CC CNRM
Train PhiNet to classify MRI as T1, T2, FLAIR
''' 
import os
import numpy as np
import shutil
import sys
from sklearn.utils import shuffle
from datetime import datetime
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.models import load_model
from models.phinet import phinet
from utils.utils import load_data, preprocess_dir, parse_args, now

if __name__ == '__main__':

    ############### DIRECTORIES ###############

    results = parse_args("train")

    TRAIN_DIR = os.path.abspath(os.path.expanduser(results.TRAIN_DIR))
    REORIENT_SCRIPT_PATH = os.path.join("utils", "reorient.sh")
    ROBUSTFOV_SCRIPT_PATH = os.path.join("utils", "robustfov.sh")
    WEIGHT_DIR = os.path.abspath(os.path.expanduser(results.OUT_DIR))

    task = results.task.lower()
    WEIGHT_DIR = os.path.join(WEIGHT_DIR, task)

    PREPROCESSED_DIR = os.path.join(TRAIN_DIR, "preprocess", task)
    if not os.path.exists(PREPROCESSED_DIR):
        os.makedirs(PREPROCESSED_DIR)

    if task == "modality":
        TASK_DIR = os.path.join(TRAIN_DIR, "modality")
    elif task == "t1-contrast":
        TASK_DIR = os.path.join(TRAIN_DIR, "t1-contrast")
    elif task == "fl-contrast":
        TASK_DIR = os.path.join(TRAIN_DIR, "fl-contrast")
    else:
        print("Invalid task")
        sys.exit()


    ############### PREPROCESSING ###############

    preprocess_dir(TASK_DIR, PREPROCESSED_DIR, REORIENT_SCRIPT_PATH, ROBUSTFOV_SCRIPT_PATH)

    ############### DATA IMPORT ###############

    X, y, filenames = load_data(PREPROCESSED_DIR)
    X, y = shuffle(X, y, random_state=0)


    num_classes = len(y[0])

    img_shape = X[0].shape
    print("Finished data processing")

    ############### MODEL SELECTION ###############

    LR = 1e-6
    LOAD_WEIGHTS = False

    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)

    if LOAD_WEIGHTS:
        weight_files = os.listdir(WEIGHT_DIR)
        weight_files.sort()
        best_weights = os.path.join(WEIGHT_DIR, weight_files[-1])
        model = load_model(best_weights)
    else:
        model = phinet(input_shape=img_shape, n_classes=num_classes, learning_rate=LR)

    ############### CALLBACKS ###############

    callbacks_list = []

    # Checkpoint
    fpath = os.path.join(
        WEIGHT_DIR, task+"_"+now()+"-epoch-{epoch:04d}-val_acc-{val_acc:.4f}.hdf5")
    checkpoint = ModelCheckpoint(
        fpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list.append(checkpoint)

    # Dynamic Learning Rate
    dlr = ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=5,
                            mode='max', verbose=1, cooldown=5, min_lr=1e-8)
    callbacks_list.append(dlr)

    # Early Stopping, used to quantify convergence
    # convergence is defined as no improvement by 1e-4 for 10 consecutive epochs
    es = EarlyStopping(monitor='loss', min_delta=0, patience=10)
    callbacks_list.append(es)

    ############### TRAINING ###############
    # the number of epochs is set high so that EarlyStopping can be the terminator
    NB_EPOCHS = 10000000
    BATCH_SIZE = 2 

    model.fit(X, y, epochs=NB_EPOCHS, validation_split=0.2,
              batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)

    #shutil.rmtree(PREPROCESSED_DIR)
    K.clear_session()
