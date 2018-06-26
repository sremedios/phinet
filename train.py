'''
Samuel Remedios
NIH CC CNRM
Train PhiNet to classify MRI as T1, T2, FLAIR
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
from models.phinet import phinet

from utils.image_generator import DataGenerator

from utils.load_data import load_data
from utils.preprocess import preprocess_dir
from utils.utils import parse_args, now

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    ############### DIRECTORIES ###############

    results = parse_args("train")
    if results.GPUID == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    WEIGHT_DIR = os.path.abspath(os.path.expanduser(results.OUT_DIR))
    PREPROCESSED_DIR = os.path.join(TRAIN_DIR, "preprocess")

    if not os.path.exists(PREPROCESSED_DIR):
        os.makedirs(PREPROCESSED_DIR)

    ############### PREPROCESSING ###############

    classes = results.classes.replace(" ","").split(',')
    class_encodings = {}
    for i, c in enumerate(classes):
        class_encodings[c] = i
        

    preprocess_dir(TRAIN_DIR, PREPROCESSED_DIR,
                   REORIENT_SCRIPT_PATH, ROBUSTFOV_SCRIPT_PATH,
                   classes,
                   results.numcores)


    ############### DATA IMPORT ###############

    #X, y, filenames, num_classes, img_shape = load_data(PREPROCESSED_DIR, classes)
    params = {'dim': (128,128,128),
            'batch_size': 8,
            'n_classes': 3,
            'class_encodings': class_encodings,
            'n_channels': 1,
            'shuffle': True}
    partition = {'val': [], 'train': []}
    sub_dirs = [os.path.join(PREPROCESSED_DIR, x) for x in os.listdir(PREPROCESSED_DIR)]
    for c in sub_dirs:
        filenames = [os.path.join(c, x) for x in os.listdir(c)]
        split = int(len(filenames) * 0.2)
        partition['val'].extend(filenames[:split])
        partition['train'].extend(filenames[split:])

    train_gen = DataGenerator(partition['train'], **params)
    val_gen = DataGenerator(partition['val'], **params)

    print("Finished data processing")

    ############### MODEL SELECTION ###############

    LR = 1e-3
    LOAD_WEIGHTS = False
    MODEL_NAME = "phinet_model_" + "-".join(results.classes.split(","))
    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME+".json")

    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)

    if LOAD_WEIGHTS:
        weight_files = os.listdir(WEIGHT_DIR)
        weight_files.sort()
        best_weights = os.path.join(WEIGHT_DIR, weight_files[-1])
        with open(MODEL_PATH) as json_data:
            model = model_from_json(json.load(json_data))
        model.load_weights(best_weights)
    else:
        model = phinet(n_classes=params['n_classes'], learning_rate=LR)

    # save model architecture to file
    json_string = model.to_json()
    with open(MODEL_PATH,'w') as f:
        json.dump(json_string, f)

    ############### CALLBACKS ###############

    callbacks_list = []

    # Checkpoint
    WEIGHT_NAME = MODEL_NAME.replace("model","weights") + "_" + now()+"-epoch-{epoch:04d}-val_acc-{val_acc:.4f}.hdf5"
    fpath = os.path.join(WEIGHT_DIR, WEIGHT_NAME)
    checkpoint = ModelCheckpoint(fpath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    callbacks_list.append(checkpoint)

    # Dynamic Learning Rate
    dlr = ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=5,
                            mode='max', verbose=1, cooldown=5, min_lr=1e-8)
    callbacks_list.append(dlr)

    # Early Stopping, used to quantify convergence
    # convergence is defined as no improvement by 1e-4 for 10 consecutive epochs
    #es = EarlyStopping(monitor='loss', min_delta=0, patience=10)
    #es = EarlyStopping(monitor='loss', min_delta=1e-8, patience=10)
    # The code continues even if the validation/training accuracy reaches 1, but loss is not.
    # For a classification task, accuracy is more important. For a regression task, loss
    # is important
    es = EarlyStopping(monitor='acc', min_delta=1e-8, patience=20)
    callbacks_list.append(es)

    ############### TRAINING ###############
    # the number of epochs is set high so that EarlyStopping can be the terminator
    NB_EPOCHS = 10000000
    #BATCH_SIZE = 2
    BATCH_SIZE = 64 # 10 should fit in a 12GB card

    '''
    model.fit(X, y, epochs=NB_EPOCHS, validation_split=0.2,
              batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)
    '''
    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        epochs=NB_EPOCHS,
                        callbacks=callbacks_list,
                        use_multiprocessing=True,
                        workers=4)

    # shutil.rmtree(PREPROCESSED_DIR)
    K.clear_session()
