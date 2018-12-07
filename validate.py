'''
Samuel Remedios
NIH CC CNRM
Predict contrast of an image.
'''

import os
import sys
from tqdm import tqdm
import time
import shutil
import json
from operator import itemgetter
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle

from utils.load_data import load_data, load_image, load_slice_data, load_nonzero_slices
from utils.utils import now, parse_args, get_classes, record_results
from utils.preprocess import preprocess_dir
from utils.patch_ops import load_patch_data

from keras.models import load_model, model_from_json
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    ############### DIRECTORIES ###############

    results = parse_args("validate")
    NUM_GPUS = 1
    if results.GPUID == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif results.GPUID == -1:
        NUM_GPUS = 3
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)

    VAL_DIR = os.path.abspath(os.path.expanduser(results.VAL_DIR))
    CUR_DIR = os.path.abspath(
        os.path.expanduser(
            os.path.dirname(__file__)
        )
    )

    PREPROCESSED_DIR = os.path.join(VAL_DIR, "preprocess")
    if not os.path.exists(PREPROCESSED_DIR):
        os.makedirs(PREPROCESSED_DIR)

    ############### MODEL SELECTION ###############

    with open(results.model) as json_data:
        model = model_from_json(json.load(json_data))
    model.load_weights(results.weights)

    ############### PREPROCESSING ###############

    classes = results.classes.replace(" ", "").split(',')

    preprocess_dir(VAL_DIR,
                   PREPROCESSED_DIR,
                   classes,
                   results.numcores)

    # get class encodings
    class_encodings = get_classes(classes)
    print(class_encodings)

    ############### DATA IMPORT ###############

    inv_class_encodings = {v: k for k, v in class_encodings.items()}

    class_directories = [os.path.join(PREPROCESSED_DIR, x)
                         for x in os.listdir(PREPROCESSED_DIR)]
    class_directories.sort()
    all_filenames = []
    for class_directory in class_directories:
        if not os.path.basename(class_directory) in classes:
            continue
        for filename in os.listdir(class_directory):
            filepath = os.path.join(class_directory, filename)
            all_filenames.append(filepath)

    ############### PREDICT ###############

    # track overall accuracy
    acc_count = len(set(all_filenames))
    unsure_count = 0
    total = len(set(all_filenames))
    total_sure_only = len(set(all_filenames))

    PRED_DIR = results.OUT_DIR
    if not os.path.exists(PRED_DIR):
        os.makedirs(PRED_DIR)
    BATCH_SIZE = 2**5

    for filename in tqdm(all_filenames):
        X = load_nonzero_slices(filename)
        y = inv_class_encodings[filename.split(os.sep)[-2]]

        # make predictions with best weights and save results
        preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0)


        # keep only "sure" predictions
        surety_threshold = .7
        try:
            preds = preds[np.where(
                np.abs(np.max(preds, axis=1) - np.min(preds, axis=1)) > surety_threshold)]
            # average over all predicted slices
            pred = np.mean(preds, axis=0)
        except:
            print("Error predicting on {}".format(filename))
            continue

        ############### RECORD RESULTS ###############
        # mean of all values must be above this value

        with open(os.path.join(PRED_DIR, "results.txt"), 'a') as f:
            with open(os.path.join(PRED_DIR, "results_errors.txt"), 'a') as e:
                # find class of prediction via max
                max_idx, max_val = max(enumerate(pred), key=itemgetter(1))
                pos = class_encodings[max_idx]

                # record confidences
                confidences = ", ".join(
                    ["{:>5.2f}".format(x*100) for x in pred])

                if max_idx == y:
                    f.write("CORRECT for {:<10} with {:<50}".format(
                        pos, filename))
                else:
                    f.write("INCRRCT for {:<10} {:<50}".format(
                        pos, filename))
                    e.write("{:<10}\t{:<50}".format(pos, filename))
                    e.write("Confidences: {}\n".format(confidences))
                    acc_count -= 1

                f.write("Confidences: {}\n".format(confidences))

                '''
                surety = np.max(pred) - np.min(pred)

                # check for surety
                if surety < surety_threshold:
                    pos = "??"  # unknown
                    f.write("UNSURE for {:<10} with {:<50}".format(
                        pos, filename))
                    unsure_count += 1
                    total_sure_only -= 1
                    acc_count -= 1

                    f.write("{:<10}\t{:<50}".format(pos, filename))
                    confidences = ", ".join(
                        ["{:>5.2f}".format(x*100) for x in pred])
                    f.write("Confidences: {}\n".format(confidences))

                else:
                    # find class of prediction via max
                    max_idx, max_val = max(enumerate(pred), key=itemgetter(1))
                    pos = class_encodings[max_idx]

                    # record confidences
                    confidences = ", ".join(
                        ["{:>5.2f}".format(x*100) for x in pred])

                    if max_idx == y:
                        f.write("CORRECT for {:<10} with {:<50}".format(
                            pos, filename))
                    else:
                        f.write("INCRRCT for {:<10} {:<50}".format(
                            pos, filename))
                        e.write("{:<10}\t{:<50}".format(pos, filename))
                        e.write("Confidences: {}\n".format(confidences))
                        acc_count -= 1

                    f.write("Confidences: {}\n".format(confidences))
                '''

    with open(os.path.join(PRED_DIR, "results.txt"), 'a') as f:
        f.write("{} of {} images correctly classified.\n"
                "Unsure Number: {}\n"
                "Accuracy: {:.2f}\n"
                "Accuracy Excluding Unsure: {:.2f}"
                .format(str(acc_count),
                        str(total),
                        str(unsure_count),
                        acc_count/total * 100.,
                        acc_count/total_sure_only * 100.,))

    print("{} of {} images correctly classified.\n"
          "Accuracy: {:.2f}\n".format(str(acc_count),
                                      str(total),
                                      acc_count/total * 100.))

    # prevent small crash from TensorFlow/Keras session close bug
    K.clear_session()
