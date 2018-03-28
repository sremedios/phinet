'''
Samuel Remedios
NIH CC CNRM
Predict contrast of an image.
'''

import os
import sys
import time
import shutil
from operator import itemgetter
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from models.phinet import phinet
from utils.utils import load_data, now, parse_args, preprocess_dir, get_classes, load_image
from keras.models import load_model
from keras import backend as K

############### DIRECTORIES ###############

results = parse_args("validate")

VAL_DIR = os.path.abspath(os.path.expanduser(results.VAL_DIR))
REORIENT_SCRIPT_PATH = os.path.join("utils", "reorient.sh")
ROBUSTFOV_SCRIPT_PATH = os.path.join("utils", "robustfov.sh")

task = results.task.lower()

PREPROCESSED_DIR = os.path.join(VAL_DIR, "preprocess", task)
if not os.path.exists(PREPROCESSED_DIR):
    os.makedirs(PREPROCESSED_DIR)

TASK_DIR = os.path.join(VAL_DIR, task)

model = load_model(results.model)

############### PREPROCESSING ###############

preprocess_dir(TASK_DIR, PREPROCESSED_DIR, REORIENT_SCRIPT_PATH, ROBUSTFOV_SCRIPT_PATH)

# get class encodings
class_encodings = get_classes(task)

############### DATA IMPORT ###############

X, y, filenames = load_data(PREPROCESSED_DIR)
num_classes = len(y[0])

print("Test data loaded.")

############### PREDICT ###############

PRED_DIR = os.path.join("validation_results")
if not os.path.exists(PRED_DIR):
    os.makedirs(PRED_DIR)
BATCH_SIZE = 16 

# make predictions with best weights and save results
preds = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

# track overall accuracy
acc_count = len(set(filenames))
total = len(set(filenames))

############### RECORD RESULTS ###############

with open(os.path.join(PRED_DIR, now()+"_results.txt"), 'w') as f:
    with open(os.path.join(PRED_DIR, now()+"_results_errors.txt"), 'w') as e:
        for filename, pred, ground_truth in zip(filenames, preds, y):

            # find class of prediction via max
            max_idx, max_val = max(enumerate(pred), key=itemgetter(1))
            max_true, val_true = max(
                enumerate(ground_truth), key=itemgetter(1))
            pos = class_encodings[max_idx]

            # record confidences
            confidences = ", ".join(["{:>5.2f}".format(x*100) for x in pred])

            if max_idx == max_true:
                f.write("CORRECT for {:<10} with {:<50}".format(pos, filename))
            else:
                f.write("INCRRCT guess with {:<10} {:<50}".format(
                    pos, filename))
                e.write("{:<10}\t{:<50}".format(pos, filename))
                e.write("Confidences: {}\n".format(confidences))
                acc_count -= 1

            f.write("Confidences: {}\n".format(confidences))
        f.write("{} of {} images correctly classified.\nAccuracy: {:.2f}\n".format(
            str(acc_count),
            str(total),
            acc_count/total * 100.))

# prevent small crash from TensorFlow/Keras session close bug
K.clear_session()
