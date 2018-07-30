'''
Samuel Remedios
NIH CC CNRM
Train PhiNet to classify MRI modalities
'''
import os
import numpy as np
import shutil
import sys
import json
from sklearn.utils import shuffle
from datetime import datetime
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from models.phinet import phinet, phinet2D
from models.multi_gpu import ModelMGPU

from utils.image_generator import DataGenerator
from utils.simple_gen import generator

from utils.load_data import load_data, load_slice_data
from utils.patch_ops import load_patch_data
from utils.preprocess import preprocess_dir
from utils.utils import parse_args, now

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    ############### DIRECTORIES ###############

    results = parse_args("train")
    NUM_GPUS = 1
    if results.GPUID == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif results.GPUID == -1:
        NUM_GPUS = 3
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)

    TRAIN_DIR = os.path.abspath(os.path.expanduser(results.TRAIN_DIR))
    CUR_DIR = os.path.abspath(
        os.path.expanduser(
            os.path.dirname(__file__)
        )
    )
    REORIENT_SCRIPT_PATH = os.path.join(CUR_DIR, "utils", "reorient.sh")
    ROBUSTFOV_SCRIPT_PATH = os.path.join(CUR_DIR, "utils", "robustfov.sh")

    classes = results.classes.replace(" ", "").split(',')

    WEIGHT_DIR = os.path.abspath(os.path.expanduser(results.OUT_DIR))
    PREPROCESSED_DIR = os.path.join(TRAIN_DIR, "preprocess")

    MODEL_NAME = "phinet_model_" + "-".join(classes)
    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME+".json")

    for d in [WEIGHT_DIR, PREPROCESSED_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    ############### PREPROCESSING ###############

    preprocess_dir(TRAIN_DIR,
                   PREPROCESSED_DIR,
                   REORIENT_SCRIPT_PATH,
                   ROBUSTFOV_SCRIPT_PATH,
                   classes,
                   results.numcores)

    ############### MODEL SELECTION ###############

    LR = 1e-4

    if results.model:
        model = load_model(results.model)
        model.load_weights(results.weights)
    else:
        model = phinet(model_path=MODEL_PATH,
                       n_classes=len(classes),
                       learning_rate=LR,
                       num_channels=1,
                       num_gpus=NUM_GPUS)

    ############### DATA IMPORT ###############

    # X, y, filenames, num_classes, img_shape = load_slice_data(PREPROCESSED_DIR,
        # classes=classes,)

    patch_size = tuple([int(x) for x in results.patch_size.split('x')])
    X, y, filenames, num_classes, img_shape = load_patch_data(PREPROCESSED_DIR,
                                                              patch_size=patch_size,
                                                              num_patches=results.num_patches,
                                                              classes=classes,)

    ############### CALLBACKS ###############

    callbacks_list = []

    # Checkpoint
    WEIGHT_NAME = MODEL_NAME.replace("model", "weights") + "_" +\
        now()+"-epoch-{epoch:04d}-val_acc-{val_acc:.4f}.hdf5"
    fpath = os.path.join(WEIGHT_DIR, WEIGHT_NAME)
    checkpoint = ModelCheckpoint(fpath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    callbacks_list.append(checkpoint)

    # Early Stopping, used to quantify convergence
    es = EarlyStopping(monitor='val_acc', min_delta=1e-8, patience=20)
    callbacks_list.append(es)

    ############### TRAINING ###############
    # the number of epochs is set high so that EarlyStopping can be the terminator
    NB_EPOCHS = 10000000
    BATCH_SIZE = 2**11

    model.fit(X, y,
              epochs=NB_EPOCHS,
              validation_split=0.2,
              batch_size=BATCH_SIZE,
              verbose=1,
              callbacks=callbacks_list)

    # shutil.rmtree(PREPROCESSED_DIR)
    K.clear_session()
