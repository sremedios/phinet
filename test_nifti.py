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
    progbar_length = 10

    ########## DIRECTORY SETUP ##########

    if len(sys.argv) < 2:
        print("Error: missing filename argument")
        sys.exit()
    FILENAMES_FILE = sys.argv[1]
    with open(FILENAMES_FILE, 'r') as f:
        filenames_labels = [l.strip().split(',') for l in f.readlines()]

    GPUID = sys.argv[2]
    os.environ['CUDA_VISIBLE_DEVICES'] = GPUID


    MODEL_NAME = "phinet"
    WEIGHT_DIR = os.path.join(
            "models", 
            "weights", 
            MODEL_NAME, 
    )

    RESULTS_DIR = os.path.join(
            "results",
    )            

    # files and paths
    for d in [WEIGHT_DIR, RESULTS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + ".json")
    HISTORY_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + "_history.json")

    # Actual instantiation happens for each fold
    with open(MODEL_PATH) as json_data:
        model = tf.keras.models.model_from_json(json.load(json_data))
    
    INIT_WEIGHT_PATH = os.path.join(WEIGHT_DIR, "init_weights.h5")
    print(model.summary(line_length=75))


    ######### FIVE FOLD CROSS VALIDATION #########

    FOLD_RESULTS_FILE = os.path.join(
            RESULTS_DIR, "test_metrics_on_nifti_ALL_DATA_fold_{}.txt"
    )

    TRAINED_WEIGHTS_FILENAME = os.path.join(
            WEIGHT_DIR, "best_weights_fold_{}.h5"
    )

    def prepare_data(x_filename, y_label, num_classes, target_dims):
        # split image into slices
        x = nib.load(x_filename).get_fdata()
        x_slices = get_slices(x, target_dims)
        x_slices = x_slices.astype(np.float32)

        # one-hot encoding
        y = np.zeros((num_classes,), dtype=np.uint8)
        y[y_label] = 1

        return x_slices, y

    # metrics
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
        
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
            return 

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


        test_accuracy.update_state(tf.argmax(y), tf.argmax(pred))
        test_loss.update_state(tf.reduce_sum(losses))

        return pred


    for cur_fold in range(5):
        
        # reset metrics
        test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        ######### MODEL AND CALLBACKS #########
        model.load_weights(TRAINED_WEIGHTS_FILENAME.format(cur_fold))

        # store corresponding scores
        x_name = []
        y_true = []
        y_pred = []

        num_elements = len(filenames_labels)

        ######### INFERENCE #########
        print()

        TEMPLATE = "\rInference... [{:{}<{}}] {}/{}"

        sys.stdout.write(TEMPLATE.format(
            "=" * 0, 
            '-', 
            progbar_length,
            0,
            num_elements,
        ))

        # get input file and target label
        for i, (filename, label) in tqdm(enumerate(filenames_labels), total=len(filenames_labels)):
            x, y = prepare_data(filename, int(label), num_classes, instance_size)

            pred = test_step((x, y))

            x_name.append(filename)
            y_true.append(y)
            y_pred.append(pred)

            if i%10 == 0:
                sys.stdout.write(TEMPLATE.format(
                    "=" * min(int(progbar_length*(i/num_elements)), 
                              progbar_length),
                    "-",
                    progbar_length,
                    i,
                    num_elements,
                ))
                sys.stdout.flush()


        with open(FOLD_RESULTS_FILE.format(cur_fold), 'w') as f:
            f.write("Test Accuracy: {}\nTest Loss: {}\n".format(
                test_accuracy.result(),
                test_loss.result(),
            ))

            f.write("{},{},{},{}\n".format(
                "filename",
                "true_class",
                "pred_class",
                "pred_score",
            ))

            for name, y_t, y_p in zip(x_name, y_true, y_pred):
                f.write("{},{},{},{}\n".format(
                    name,
                    np.argmax(y_t),
                    np.argmax(y_p),
                    y_p,
                ))
                

        print("\nMetrics using weights from fold {}:\n\
                Accuracy: {}\n\
                Loss:     {}\n".format(
                    cur_fold,
                    test_accuracy.result(),
                    test_loss.result()
                    )
        )
