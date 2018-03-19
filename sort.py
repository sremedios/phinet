'''
Samuel Remedios
NIH CC CNRM
Example using trained weights for automatic sorting of a directory
'''

import os
from time import time
from operator import itemgetter
import shutil
import nibabel as nib
import numpy as np

start_time = time()
from models.phinet import phinet
print("Elapsed time to load tensorflow:", time()-start_time)

from utils.preprocessing import load_data
from keras.models import load_model
from keras import backend as K

start_time = time()

############### DIRECTORIES ###############

DATA_DIR = "sorting_example"
UNSORTED_DIR = os.path.join(DATA_DIR, "unsorted")
PREPROCESSED_TMP_DIR = os.path.join(DATA_DIR, "preprocess")

# this dir must point to the last preprocessing step
PREPROCESSED_FINAL_STEP_DIR = os.path.join(PREPROCESSED_TMP_DIR, "robustfov")

WEIGHT_DIR = os.path.join("weights")

if not os.path.exists(PREPROCESSED_TMP_DIR):
    os.makedirs(PREPROCESSED_TMP_DIR)

############### DATA IMPORT ###############
PATCH_SIZE = (45, 45, 15)

X, filenames = load_data(
    UNSORTED_DIR, PREPROCESSED_TMP_DIR, PATCH_SIZE, labels_known=False)

print("Elapsed time to preprocess data:", time()-start_time)
start_time = time()

# get class encodings
class_encodings = {}
with open(os.path.join(DATA_DIR, "..", "class_encodings.txt"), 'r') as f:
    content = f.read().split('\n')
for line in content:
    if len(line) == 0:
        continue
    entry = line.split()
    class_encodings[int(entry[1])] = entry[0]

num_classes = len(class_encodings)

############### LOAD MODEL ###############

WEIGHT_INDEX = -1  # load most recent weights
weight_files = os.listdir(WEIGHT_DIR)
weight_files.sort()
best_weights = os.path.join(WEIGHT_DIR, weight_files[WEIGHT_INDEX])
model = load_model(best_weights)

############### PREDICT ###############

BATCH_SIZE = 128

# make predictions with best weights and save results
preds = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

############### AGGREGATE PATCHES ###############

# initialize aggregate
final_pred_scores = {}
for filename in set(filenames):
    final_pred_scores[filename] = np.zeros(preds[0].shape)

for pred, filename in zip(preds, filenames):
    # add the average of the score, scaled uniformly by the number of patches for that filename
    final_pred_scores[filename] += pred / filenames.count(filename)

############### SORT ###############

# some preprocessing may slightly rename the files
# this allows us to move the proper, corresponding file
processed_filenames = os.listdir(PREPROCESSED_FINAL_STEP_DIR)
processed_filenames.sort()
true_filenames = os.listdir(UNSORTED_DIR)
true_filenames.sort()

filename_mapping = {}
for processed_filename, true_filename in zip(processed_filenames, true_filenames):
    filename_mapping[processed_filename] = true_filename

# use predcitions to sort
for filename, pred in final_pred_scores.items():

    max_idx, max_val = max(enumerate(pred), key=itemgetter(1))
    pos = class_encodings[max_idx]

    dst_dir = os.path.join(DATA_DIR, pos)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # move file
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    shutil.move(os.path.join(UNSORTED_DIR, filename_mapping[filename]),
                os.path.join(dst_dir, filename_mapping[filename]))

############### DELETE TMP DIRECTORY ###############

shutil.rmtree(PREPROCESSED_TMP_DIR)

############### ELAPSED TIME ###############

print("Elapsed time:", time() - start_time)
K.clear_session()
