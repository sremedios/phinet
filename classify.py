import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
from models.phinet import *
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
import tensorflow as tf
from tqdm import tqdm
from utils.preprocess import preprocess
from scipy.stats import mode

int_to_class = {i: c for i, c in 
        enumerate(sorted(['FL', 'FLC', 'PD', 'T1', 'T1C', 'T2']))}
class_to_int = {c: i for i, c in int_to_class.items()}

def get_axial_slices(img_vol):
    tmp = np.moveaxis(img_vol, 2, 0)
    tmp = tmp[25:]
    tmp = tmp[:-25]
    tmp = np.array(tmp)[..., np.newaxis]
    return tmp

def normalize(img):
    q = np.percentile(img[np.nonzero(img)], 99)
    img[img>q] = q
    img = (img - img.min()) / (img.max() - img.min())
    return img

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def prepare_data(x_filename):
    x = nib.load(str(x_filename)).get_fdata(dtype=np.float32)
    x = get_axial_slices(x)
    x = normalize(x)

    return x

if __name__ == "__main__":

    WEIGHT_DIR = Path(sys.argv[1])
    MODEL_PATH = Path(sys.argv[2])
    fname = Path(sys.argv[3])
    RESULTS_DIR = Path(sys.argv[4])
    RESULTS_FILE = RESULTS_DIR / "classification_results.csv"
    GPUID = sys.argv[5]
    FIXED_FPATH = Path(sys.argv[6])
    ANTS_REG_PATH = Path("utils/AntsExample.sh")
    DENOISE_PATH = Path("utils/remove_background_noise.py")
    os.environ['CUDA_VISIBLE_DEVICES'] = GPUID

    ########## DIRECTORY SETUP ##########

    # files and paths
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)

    ######### MODEL #########
    with open(MODEL_PATH) as json_data:
        model = tf.keras.models.model_from_json(json.load(json_data))
    #print(model.summary(line_length=75))
    TRAINED_WEIGHTS_FILENAME = WEIGHT_DIR / "best_weights_fold_{}.h5"

    ######### APPLY PREPROCESS #########
    prep_fname = Path("preprocessed_" + fname.name)
    preprocess(fname, prep_fname, FIXED_FPATH, ANTS_REG_PATH, DENOISE_PATH)
    x = prepare_data(prep_fname)
    os.remove(prep_fname)

    ######### INFERENCE #########
    fold_preds = []
    scores = []
    for cur_fold in range(5):

        model.load_weights(str(TRAINED_WEIGHTS_FILENAME).format(cur_fold))

        logits = model(x, training=False)
        preds = tf.nn.softmax(logits)
        pred = tf.reduce_mean(preds,axis=0)
        
        scores.extend(preds)

    scores = tf.reduce_mean(scores, axis=0).numpy()
    pred = int(tf.argmax(scores).numpy())

    probability_template = "FL: {:.2%} FLC: {:.2%} PD: {:.2%} T1: {:.2%} T1C: {:.2%} T2: {:.2%}"

    with open(RESULTS_FILE, 'a') as f:
        f.write("{},{},{}\n".format(
            fname,
            int_to_class[pred],
            probability_template.format(*scores),
        ))
