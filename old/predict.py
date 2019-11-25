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
import nibabel as nib
from operator import itemgetter
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from models.phinet import phinet

from utils.load_data import load_data, load_image
from utils.patch_ops import get_patches
from utils.preprocess import preprocess
from utils.utils import now, parse_args, get_classes, record_results

from keras.models import load_model, model_from_json
from keras import backend as K
import tempfile
from keras.engine import Input, Model
os.environ['FSLOUTPUTTYPE']='NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

    ############### DIRECTORIES ###############

    results = parse_args("test")
    NUM_GPUS = 1
    if results.GPUID == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif results.GPUID == -1:
        NUM_GPUS = 3
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)


    CUR_DIR = os.path.abspath(
            os.path.expanduser(
                os.path.dirname(__file__)
                )
            )

    #TMP_DIR = tempfile.mkdtemp()
    #PREPROCESSED_DIR = tempfile.mkdtemp()
    TMP_DIR = os.path.join(CUR_DIR, "temp_stuff")
    PREPROCESSED_DIR = os.path.join(CUR_DIR, "preprocessed_stuff")

    for d in [TMP_DIR, PREPROCESSED_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    ############### MODEL SELECTION ###############

    with open(results.model) as json_data:
        model = model_from_json(json.load(json_data))
    model.load_weights(results.weights)

    ############### PREPROCESSING ###############

    classes = results.classes.replace(" ","").split(',')

    preprocess(os.path.basename(results.INFILE),
               os.path.dirname(results.INFILE),
               PREPROCESSED_DIR,
               TMP_DIR,
               verbose=0,
               remove_tmp_files=True)

    class_encodings = get_classes(classes)

    ############### PREDICT ###############

    filename = os.path.join(PREPROCESSED_DIR, os.path.basename(results.INFILE))
    img = nib.load(filename).get_data()

    patch_size = tuple([int(x) for x in results.patch_size.split('x')])
    patches = get_patches(img,
                          filename,
                          patch_size,
                          results.num_patches)

    # make predictions with best weights and save results
    preds = model.predict(patches, batch_size=2**4, verbose=1)

    ############### RECORD RESULTS ###############

    confidences = ",".join(["{:.2f}".format(x*100) for x in np.mean(preds, axis=0)])
    max_idx, max_val = max(enumerate(np.mean(preds, axis=0)), key=itemgetter(1))
    pred_class = class_encodings[max_idx]
    classids = ",".join(["Prob({})".format(x) for x in classes])
    print(classids)
    print(confidences)
    print(pred_class)

    if not os.path.exists(results.OUTFILE):
        with open(results.OUTFILE, 'w') as csvfile:
            csvfile.write("filename,prediction,%s\n" % classids )

    x = os.path.basename(filename) + "," + pred_class + "," + confidences
    with open(results.OUTFILE, 'a') as csvfile:
        csvfile.write("%s\n" % x)

    if results.clear == "y":
        shutil.rmtree(PREPROCESSED_DIR)
        shutil.rmtree(TMP_DIR)

    # prevent small crash from TensorFlow/Keras session close bug
    K.clear_session()
