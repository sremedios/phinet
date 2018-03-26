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
from utils.utils import load_data, now, parse_testing_args, preprocess_file, get_classes, load_image
from keras.models import load_model
from keras import backend as K


############### DIRECTORIES ###############

results = parse_testing_args()
model = load_model(results.model)
PREPROCESS_SCRIPT_PATH = os.path.join("utils", "preprocess.sh")

############### PREPROCESSING ###############

new_filename = preprocess_file(results.INFILE, results.OUT_DIR, PREPROCESS_SCRIPT_PATH)
class_encodings = get_classes(results.encodings_file)

############### PREDICT ###############
filename = os.path.join(results.OUT_DIR, new_filename)
X = load_image(filename)

# make predictions with best weights and save results
preds = model.predict(X, batch_size=1, verbose=1)

############### RECORD RESULTS ###############

with open(results.OUTFILE, 'w') as f:
    pred = preds[0]
    # find class of prediction via max
    max_idx, max_val = max(enumerate(pred), key=itemgetter(1))
    pos = class_encodings[max_idx]

    f.write("{:<10} {:<10}".format(os.path.basename(filename), pos))

    # record confidences
    confidences = ", ".join(["{:>5.2f}".format(x*100) for x in pred])

    f.write("Confidences: {}\n".format(confidences))

if results.clear == "y":
    shutil.rmtree(results.OUT_DIR)

# prevent small crash from TensorFlow/Keras session close bug
K.clear_session()
