'''
Samuel Remedios
NIH CC CNRM
Predict contrast of an image.
'''

import os
import time
import shutil
from operator import itemgetter
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from models.phinet import phinet
from utils.load_data import load_data_multiclass, preprocess_training_data
from keras.models import load_model
from keras import backend as K

############### DIRECTORIES ###############

MODE = "multiclass"
MODEL_VERSION = "phinet"
MODEL_NAME = MODEL_VERSION + "_" + MODE
WEIGHT_DIR = os.path.join("weights", MODEL_NAME)

VAL_DIR = os.path.join('data', 'validation')
PREPROCESSED_TMP_DIR = os.path.join("tmp_processing")

############## DATA PREPROCESSING ###############

preprocess_training_data(VAL_DIR, PREPROCESSED_TMP_DIR)

############### DATA IMPORT ###############

def now():
    return datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")

X, labels, filenames = load_data_multiclass(
    PREPROCESSED_TMP_DIR, test=True, model_name=MODEL_NAME)
INPUT_SHAPE = X[0].shape
print("Test data loaded.")

############### LOAD MODEL ###############

# load best weights
WEIGHT_INDEX = -1  # -1 implies most recent, -2 implies second most recent, and so on
weight_files = os.listdir(WEIGHT_DIR)
weight_files.sort()
best_weights = os.path.join(WEIGHT_DIR, weight_files[WEIGHT_INDEX])
model = load_model(best_weights)

PRED_DIR = os.path.join(VAL_DIR, "predictions", MODEL_NAME)
if not os.path.exists(PRED_DIR):
    os.makedirs(PRED_DIR)
BATCH_SIZE = 32

# make predictions with best weights and save results
preds = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
i = 0
acc_count = len(filenames)
total = len(filenames)

with open(os.path.join(PRED_DIR, now()+"_production_preds.txt"), 'w') as f:
    with open(os.path.join(PRED_DIR, now()+"_production_errors.txt"), 'w') as e:
        for line, trueLabel in zip(preds, labels):

            # find class of prediction via max
            max_idx, max_val = max(enumerate(line), key=itemgetter(1))
            max_true, val_true = max(enumerate(trueLabel), key=itemgetter(1))

            if max_idx == 0:
                pos = "T1"
            elif max_idx == 1:
                pos = "T2"
            else:
                pos = "FL"

            if max_idx == max_true:
                f.write("CORRECT for " + pos + " with\t" + filenames[i])
            else:
                f.write("INCRRCT guess with " + pos + "\t" + filenames[i])
                e.write(pos + '\t' + filenames[i])
                e.write(
                    "\tConfidences: " + str(line[0]) + ", " +
                    str(line[1]) + "," + str(line[2])+'\n')
                acc_count -= 1

            f.write("\tConfidences: " +
                    str(line[0]) + ", " + str(line[1]) + "," + str(line[2]))
            f.write('\n')
            i += 1
        f.write(str(acc_count) + " out of " +
                str(total) + " non-robex images.\n")
        f.write("Accuracy: " + str(acc_count/total * 100.) + '\n')

shutil.rmtree(PREPROCESSED_TMP_DIR)

# prevent small crash from TensorFlow/Keras session close bug
K.clear_session()
