'''

NOTE: for now, images MUST be preprocessed ahead of time

IE: the filenames file provided must point to preprocessed images

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from pathlib import Path
import json

import numpy as np
import nibabel as nib

import tensorflow as tf
from tqdm import tqdm

from utils import preprocess
from models.phinet import *

int_to_class = {i: c for i, c in 
        enumerate(sorted(['FL', 'FLC', 'PD', 'T1', 'T1C', 'T2']))}
class_to_int = {c: i for i, c in int_to_class.items()}


def normalize(img):
    q = np.percentile(img[np.nonzero(img)], 99)
    img[img>q] = q
    img = (img - img.min()) / (img.max() - img.min())
    return img

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def prepare_data(x_filename, y_label, num_classes):
    x = nib.load(str(x_filename)).get_fdata(dtype=np.float32)
    x = normalize(x)

    y = np.array(tf.one_hot(y_label, depth=num_classes), dtype=np.float32) 
    return x[np.newaxis, ..., np.newaxis], y

if __name__ == "__main__":

    WEIGHT_DIR = Path(sys.argv[1])
    FNAMES_FILE = sys.argv[2]
    cur_fold = sys.argv[3]
    GPUID = sys.argv[4]
    os.environ['CUDA_VISIBLE_DEVICES'] = GPUID

    ########## HYPERPARAMETER SETUP ##########

    num_classes = 6
    progbar_length = 10

    ########## DIRECTORY SETUP ##########

    with open(FNAMES_FILE, 'r') as f:
        lines = [l.strip().split(',') for l in f.readlines()]
    filenames_labels = [(Path(s).resolve(), int(float(l))) for (s, l) in lines]
    # strip accidental duplicates
    filenames_labels = sorted(set(filenames_labels))


    MODEL_NAME = WEIGHT_DIR.name
    RESULTS_DIR = Path("results") / MODEL_NAME

    # files and paths
    for d in [WEIGHT_DIR, RESULTS_DIR]:
        if not d.exists():
            d.mkdir(parents=True)

    MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")

    ######### MODEL #########
    with open(MODEL_PATH) as json_data:
        model = tf.keras.models.model_from_json(json.load(json_data))
    #print(model.summary(line_length=75))
    TRAINED_WEIGHTS_FILENAME = WEIGHT_DIR / "best_weights_fold_{}.h5"
    model.load_weights(str(TRAINED_WEIGHTS_FILENAME).format(cur_fold))

    ######### GRADCAM PREPARATION ######### 

    FOLD_RESULTS_FILE = RESULTS_DIR / "test_preds_fold_{}.csv"
    
    # metrics
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # store corresponding scores
    x_name = []
    y_true = []
    y_pred = []

    num_elements = len(filenames_labels)

    ######### INFERENCE #########
    print()

    # write head
    with open(str(FOLD_RESULTS_FILE).format(cur_fold), 'w') as f:
        f.write("{},{},{},{}\n".format(
            "filename",
            "true_class",
            "pred_class",
            "pred_score",
        ))

    # get input file and target label
    for i, (filename, label) in tqdm(enumerate(filenames_labels), 
            total=len(filenames_labels)):

        x, y = prepare_data(filename, label, len(class_to_int))

        logits = model(x, training=False)

        pred = tf.nn.softmax(logits)[0]

        with open(str(FOLD_RESULTS_FILE).format(cur_fold), 'a') as f:
            # write current predictions
            f.write("{},{},{}".format(
                filename,
                np.argmax(y),
                np.argmax(pred),
            ))
            f.write(",[")
            for class_pred in pred: 
                f.write("{:.2f} ".format(class_pred))
            f.write("]\n")
