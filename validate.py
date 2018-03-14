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
from utils.preprocessing import load_data
from keras.models import load_model
from keras import backend as K


def now():
    return datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")

############### DIRECTORIES ###############


VAL_DIR = os.path.join('data', 'validation')
PREPROCESSED_TMP_DIR = os.path.join("data", "robustfov")
WEIGHT_DIR = os.path.join("weights")

############### DATA IMPORT ###############

PATCH_SIZE = (45, 45, 15)

X, y, filenames = load_data(VAL_DIR, PREPROCESSED_TMP_DIR, PATCH_SIZE)
num_classes = len(y[0])

# get class encodings
class_encodings = {}
with open(os.path.join(VAL_DIR, "..", "class_encodings.txt"), 'r') as f:
    content = f.read().split('\n')
for line in content:
    if len(line) == 0:
        continue
    entry = line.split()
    class_encodings[int(entry[1])] = entry[0]

print("Test data loaded.")

############### LOAD MODEL ###############

# load best weights
WEIGHT_INDEX = -1  # -1 implies most recent, -2 implies second most recent, and so on
weight_files = os.listdir(WEIGHT_DIR)
weight_files.sort()
best_weights = os.path.join(WEIGHT_DIR, weight_files[WEIGHT_INDEX])
model = load_model(best_weights)

############### PREDICT ###############

PRED_DIR = os.path.join("validation_results")
if not os.path.exists(PRED_DIR):
    os.makedirs(PRED_DIR)
BATCH_SIZE = 128

# make predictions with best weights and save results
preds = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

# track overall accuracy
acc_count = len(set(filenames))
total = len(set(filenames))

############### AGGREGATE PATCHES ###############

# initialize aggregate
final_pred_scores = {}
final_ground_truth = {}
for filename in set(filenames):
    final_pred_scores[filename] = np.zeros(preds[0].shape)

for pred, true_label, filename in zip(preds, y, filenames):
    # this will get overwritten each time, but the ground truth is always the ground truth
    final_ground_truth[filename] = true_label

    # add the average of the score, scaled uniformly by the number of patches for that filename
    final_pred_scores[filename] += pred / filenames.count(filename)


############### RECORD RESULTS ###############

with open(os.path.join(PRED_DIR, now()+"_results.txt"), 'w') as f:
    with open(os.path.join(PRED_DIR, now()+"_results_errors.txt"), 'w') as e:
        for filename, pred in final_pred_scores.items():

            # find class of prediction via max
            max_idx, max_val = max(enumerate(pred), key=itemgetter(1))
            max_true, val_true = max(
                enumerate(final_ground_truth[filename]), key=itemgetter(1))
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


shutil.rmtree(PREPROCESSED_TMP_DIR)

# prevent small crash from TensorFlow/Keras session close bug
K.clear_session()
