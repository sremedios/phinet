import numpy as np
import os
import sys
import json

import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

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
            RESULTS_DIR, "test_metrics_fold_{}.h5"
    )

    TRAINED_WEIGHTS_FILENAME = os.path.join(
            WEIGHT_DIR, "best_weights_fold_{}.h5"
    )

    TEST_TF_RECORD_FILENAME = os.path.join(
            "data", "tfrecord_dir", "dataset_fold___test.tfrecord"
    )

    for cur_fold in range(5):

        ######### MODEL AND CALLBACKS #########
        model.load_weights(TRAINED_WEIGHTS_FILENAME.format(cur_fold))

        ######### DATA IMPORT #########

        test_dataset = tf.data.TFRecordDataset(
                TEST_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_into_volume(
                record,
                instance_size,
                num_labels=num_classes))\

        num_elements = 2072 # known ahead of time from TFRecord creation

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

            if logits.shape != y.shape:
                print("Shape mismatch, skipping")
                break

            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=y,
                logits=logits,
            )

            # Aggregation of scores
            # For now, just take maximum class
            pred = tf.reduce_sum(
                tf.nn.softmax(logits),
                axis=0,
            )

            test_accuracy.update_state(y, pred)
            test_loss.update_state(tf.reduce_sum(losses))

    
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

        # Inference
        for i, data in enumerate(test_dataset):
            test_step(data)

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
            f.write("Test Accuracy: {}\nTest Loss: {}".format(
                test_accuracy.result(),
                test_loss.result(),
            ))

        print("Metrics using weights from fold {}:\n\
                Accuracy: {}\n\
                Loss:     {}\n".format(
                    test_accuracy.result(),
                    test_loss.result()
                    )
        )
