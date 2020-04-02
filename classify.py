'''

NOTE: for now, images MUST be preprocessed ahead of time

IE: the filenames file provided must point to preprocessed images

'''
import numpy as np
import os
import sys
import json

import nibabel as nib

import tensorflow as tf
from tqdm import tqdm

from utils.pad import *
from utils import preprocess
from utils.patch_ops import *
from utils.augmentations import *
from utils.tfrecord_utils import *
from models.phinet import *

def prepare_data(x_filename, y_label, num_classes, target_dims):
    # split image into slices
    x = nib.load(x_filename).get_fdata()
    x_slices = get_slices(x, target_dims)
    x_slices = x_slices.astype(np.float32)

    # one-hot encoding
    y = np.zeros((num_classes,), dtype=np.uint8)
    y[y_label] = 1

    return x_slices, y

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def mil_prediction(pred, n=1):
    # generalized, returns top n 
    i = tf.argsort(pred[:, 1], axis=0)
    i = i[len(pred) - n : len(pred)]

    return (tf.gather(pred, i), i)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    ########## HYPERPARAMETER SETUP ##########

    instance_size = (256, 256)
    num_classes = 6

    ########## DIRECTORY SETUP ##########

    if len(sys.argv) < 2:
        print("Error: missing filename argument")
        sys.exit()

    fname = sys.argv[1]

    GPUID = sys.argv[2]
    os.environ['CUDA_VISIBLE_DEVICES'] = GPUID

    MODEL_NAME = "phinet"
    WEIGHT_DIR = os.path.join(
            "models", 
            "weights", 
            MODEL_NAME, 
    )

    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + ".json")

    # Actual instantiation happens for each fold
    with open(MODEL_PATH) as json_data:
        model = tf.keras.models.model_from_json(json.load(json_data))
    

    ######### FIVE FOLD CROSS VALIDATION #########


        
    # step
    def test_step(inputs):
        x, y = inputs

        logits = tf.map_fn(
            lambda cur_slice: model(
                tf.reshape(
                    cur_slice, 
                    (1,) + tuple(cur_slice.shape.as_list())
                ), 
                training=False
            ),
            x,
        )

        if logits.shape[-1] != y.shape[-1]:
            print("\nShape mismatch.\nLogits : {}\nY      : {}".format(
                    logits.shape,
                    y.shape
                )
            )
            return -1.0 * tf.ones_like(y, dtype=tf.float32)

        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=y,
            logits=logits,
        )


        # Aggregation of scores
        # Majority vote === mean of all slices

        
        pred = tf.reduce_mean(
            tf.nn.softmax(logits),
            axis=0,
        )[0]


        test_accuracy.update_state([tf.argmax(y)], [tf.argmax(pred)])
        test_loss.update_state(tf.reduce_sum(losses))

        return pred


    class_mapping = {
            0: 'FL',
            1: 'FLC',
            2: 'PD',
            3: 'T1',
            4: 'T1C',
            5: 'T2',
        }

    ######### MODEL AND CALLBACKS #########

    preds = []
    for cur_fold in range(5):
        TRAINED_WEIGHTS_FILENAME = os.path.join(WEIGHT_DIR, "best_weights_fold_{}.h5")
        model.load_weights(TRAINED_WEIGHTS_FILENAME.format(cur_fold))

        x, y = prepare_data(fname, 0, num_classes, instance_size)
        print(x.shape, y.shape)
        pred = test_step((x, y))
        preds.append(pred)


    ######### INFERENCE #########
    print()

    print(class_mapping)
    print("Filename: {}\nPrediction: {}\nConfidences: {}".format(
        fname,
        class_mapping[int(tf.argmax(tf.reduce_mean(preds, axis=1)).numpy())],
        tf.reduce_mean(preds, axis=1),
    ))

