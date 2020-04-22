import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import sys
import json
import time

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tqdm import tqdm

from utils.augmentations import *
from utils.tfrecord_utils import *
from utils.pad import *
from utils.progbar import show_progbar
from models.phinet import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n


if __name__ == "__main__":

    #policy = mixed_precision.Policy('mixed_float16')
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_policy(policy)
    loss_scale = policy.loss_scale

    DATA_DIR = Path(sys.argv[1])
    cur_fold = int(sys.argv[2])
    gpuid = sys.argv[3]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

    ########## HYPERPARAMETER SETUP ##########

    N_EPOCHS = 10000
    BATCH_SIZE = 2**3
    BUFFER_SIZE = BATCH_SIZE * 2
    ds = 4
    num_classes = 6
    learning_rate = 1e-4
    progbar_length = 10
    CONVERGENCE_EPOCH_LIMIT = 10
    TERMINATING_EPOCH = 500 # stop training at 50 epochs
    epsilon = 1e-3

    TRAIN_COLOR_CODE = "\033[0;32m"
    VAL_COLOR_CODE = "\033[0;36m"

    ########## DIRECTORY SETUP ##########

    MODEL_NAME = "phinet_3d"
    WEIGHT_DIR = Path("models/weights") / MODEL_NAME
    RESULTS_DIR = Path("results") / MODEL_NAME

    # files and paths
    for d in [WEIGHT_DIR, RESULTS_DIR]:
        if not d.exists():
            d.mkdir(parents=Path('.'))

    MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")
    HISTORY_PATH = WEIGHT_DIR / (MODEL_NAME + "_history.json")

    # Actual instantiation happens for each fold
    model = phinet(num_classes=num_classes, ds=ds)
    #model = resnet(num_classes=num_classes, ds=ds)
    
    INIT_WEIGHT_PATH = WEIGHT_DIR / "init_weights.h5"
    model.save_weights(str(INIT_WEIGHT_PATH))
    json_string = model.to_json()
    with open(str(MODEL_PATH), 'w') as f:
        json.dump(json_string, f)

    #print(model.summary(line_length=75))


    ######### FIVE FOLD CROSS VALIDATION #########

    TRAIN_CURVE_FILENAME = RESULTS_DIR / "training_curve_fold_{}.csv"

    TRAIN_TF_RECORD_FILENAME = DATA_DIR / "dataset_fold_{}_train.tfrecord"
    VAL_TF_RECORD_FILENAME = DATA_DIR / "dataset_fold_{}_val.tfrecord"
    data_count_fname = DATA_DIR / "data_count.txt"

    data_count = {"{}_{}".format(t, fold): int(n) for t, fold, n in map(lambda l: l.strip().split(','), open(data_count_fname, 'r').readlines())}

    print("Current fold: {}".format(cur_fold))
    
    with open(str(TRAIN_CURVE_FILENAME).format(cur_fold), 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    ######### MODEL AND CALLBACKS #########
    model.load_weights(str(INIT_WEIGHT_PATH))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    ######### DATA IMPORT #########


    parse = lambda record: parse_into_volume(record)

    train_dataset = tf.data.TFRecordDataset(
            str(TRAIN_TF_RECORD_FILENAME).format(cur_fold))\
        .map(parse)\
        .shuffle(BUFFER_SIZE)\
        .batch(BATCH_SIZE)

    val_dataset = tf.data.TFRecordDataset(
            str(VAL_TF_RECORD_FILENAME).format(cur_fold))\
        .map(parse)\
        .batch(1)

    # metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
    val_accuracy = tf.keras.metrics.Accuracy(name='val_acc')
    val_loss = tf.keras.metrics.Mean(name='val_loss')


    # step
    def train_step(inputs):
        x, y = inputs
        
        # todo: fp16 doens't work yet
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_object(y, logits)
            scaled_loss = opt.get_scaled_loss(loss)

        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        grads = opt.get_unscaled_gradients(scaled_gradients)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        train_accuracy.update_state(y, tf.nn.softmax(logits))
        return loss

    def val_step(inputs):
        x, y = inputs

        logits = model(x, training=False)

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y,
            logits=logits,
        )

        val_loss.update_state(loss)

        # arithmetic mean of prediction of all slices
        pred = tf.nn.softmax(logits)
        # update val acc by taking highest class: argmax
        val_accuracy.update_state(tf.argmax(y, axis=1), tf.argmax(pred, axis=1))


    ######### TRAINING #########

    best_val_loss = 100000
    best_val_acc = 0
    convergence_epoch_counter = 0


    best_epoch = 1
    for cur_epoch in range(N_EPOCHS):

        cur_step = 1
        iterator = iter(train_dataset)
        elapsed_batch_time = 0.0
        while True:
            try:
                batch_start_time = time.time()
                data = next(iterator)
            except StopIteration:
                break
            else:
                loss = train_step(data)
                train_loss.update_state(loss)
                num_elements = data_count["{}_{}".format("train", cur_fold)]

                batch_end_time = time.time()

                elapsed_batch_time = running_average(
                        elapsed_batch_time,
                        batch_end_time - batch_start_time,
                        cur_step,
                    )

                show_progbar(
                        cur_epoch + 1,
                        N_EPOCHS,
                        BATCH_SIZE,
                        cur_step,
                        num_elements,
                        train_loss.result(),
                        train_accuracy.result(),
                        TRAIN_COLOR_CODE,
                        elapsed_batch_time,
                        progbar_length,
                    )
                cur_step += 1

        print()

        # validation metrics
        cur_step = 1
        iterator = iter(val_dataset)
        elapsed_batch_time = 0.0
        while True:
            try:
                batch_start_time = time.time()
                data = next(iterator)
            except StopIteration:
                break
            else:
                val_step(data)

                num_elements = data_count["{}_{}".format("val", cur_fold)]

                batch_end_time = time.time()
                elapsed_batch_time = running_average(
                        elapsed_batch_time,
                        batch_end_time - batch_start_time,
                        cur_step,
                    )
                show_progbar(
                        cur_epoch + 1,
                        N_EPOCHS,
                        1,
                        cur_step,
                        num_elements,
                        val_loss.result(),
                        val_accuracy.result(),
                        VAL_COLOR_CODE,
                        elapsed_batch_time,
                        progbar_length,
                    )
                cur_step += 1
        print()
        

        with open(str(TRAIN_CURVE_FILENAME).format(cur_fold), 'a') as f:
            f.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                cur_epoch + 1,
                train_loss.result().numpy(),
                train_accuracy.result().numpy(),
                val_loss.result().numpy(),
                val_accuracy.result().numpy(),
            ))


        #if convergence_epoch_counter >= CONVERGENCE_EPOCH_LIMIT:
        # Stop training at TERMINATING_EPOCH
        if cur_epoch >= TERMINATING_EPOCH:
            print("\nCurrent Fold: {}\
                    \n{} epochs have occured, model is converged.\
                    \nModel achieved best val loss at epoch {}.\
                    \nTrain Loss: {:.4f} Train Acc: {:.2%}\
                    \nVal   Loss: {:.4f} Val   Acc: {:.2%}".format(
                cur_fold,
                cur_epoch,
                best_epoch,
                train_loss.result().numpy(), 
                train_accuracy.result().numpy(),
                val_loss.result().numpy(), 
                val_accuracy.result().numpy(),
            ))
            break

        if val_loss.result() > best_val_loss and\
                np.abs(val_loss.result() - best_val_loss) > epsilon:
            convergence_epoch_counter += 1
        else:
            convergence_epoch_counter = 0

        if val_loss.result() < best_val_loss:
            best_epoch = cur_epoch + 1
            best_val_loss = val_loss.result() 
            best_val_acc = val_accuracy.result()
            model.save_weights(
                str(WEIGHT_DIR / "best_weights_fold_{}.h5".format(cur_fold))
            )
