'''
Samuel Remedios
NIH CC CNRM
Predict contrast of an image.
'''

import os
import sys
import time
import shutil
import json
from operator import itemgetter
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from models.phinet import phinet
from utils.utils import load_data, now, parse_args, preprocess, get_classes, load_image
from keras.models import load_model, model_from_json
from keras import backend as K
import tempfile
from keras.engine import Input, Model
os.environ['FSLOUTPUTTYPE']='NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

    ############### DIRECTORIES ###############

    results = parse_args("test")
    if results.GPUID == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)

    with open(results.model) as json_data:
        model = model_from_json(json.load(json_data))
    model.load_weights(results.weights)

    CUR_DIR = os.path.abspath(
            os.path.expanduser(
                os.path.dirname(__file__)
                )
            )

    REORIENT_SCRIPT_PATH = os.path.join(CUR_DIR, "utils", "reorient.sh")
    ROBUSTFOV_SCRIPT_PATH = os.path.join(CUR_DIR, "utils", "robustfov.sh")
    TMP_DIR=tempfile.mkdtemp()
    results.PREPROCESSED_DIR=tempfile.mkdtemp()

    #TMP_DIR = os.path.join(results.PREPROCESSED_DIR, "tmp_intermediate_preprocessing")
    #TMP_DIR = os.path.join(results.PREPROCESSED_DIR, X) # get a temporary directory, rather than fixed.
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    task = results.task.lower()

    ############### PREPROCESSING ###############

    new_filename = preprocess(results.INFILE,
                              results.PREPROCESSED_DIR,
                              TMP_DIR,
                              REORIENT_SCRIPT_PATH,
                              ROBUSTFOV_SCRIPT_PATH,
                              verbose=0,)

    class_encodings = get_classes(results.classes.split(','))

    ############### PREDICT ###############

    filename = os.path.join(results.PREPROCESSED_DIR, new_filename)
    X = load_image(filename)

    # make predictions with best weights and save results
    preds = model.predict(X, batch_size=1, verbose=1)

    ############### RECORD RESULTS ###############

    confidences = ";".join(["{:.2f}".format(x*100) for x in preds[0]])
    max_idx, max_val = max(enumerate(preds[0]), key=itemgetter(1))
    pred_class = class_encodings[max_idx]
    record_results(results.OUTFILE, (os.basename(filename), pred_class, confidences)
    """
    with open(results.OUTFILE, 'a') as f:
        pred = preds[0]
        # find class of prediction via max
        max_idx, max_val = max(enumerate(pred), key=itemgetter(1))
        pos = class_encodings[max_idx]

        f.write("{:<10}\t{:<10}".format(os.path.basename(filename), pos))
        print("{:<10}\t{:<10}".format(os.path.basename(filename), pos))

        # record confidences
        confidences = ", ".join(["{:>5.2f}".format(x*100) for x in pred])

        f.write("Confidences: {}\n".format(confidences))
        print("Confidences: {}\n".format(confidences))
    """


    if results.clear == "y":
        shutil.rmtree(results.PREPROCESSED_DIR)
    shutil.rmtree(TMP_DIR)

    # prevent small crash from TensorFlow/Keras session close bug
    K.clear_session()
