import numpy as np
import os
import sys
import json

from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

#from utils.augmentations import *
from utils.tfrecord_utils import *
from utils.pad import *
from models.phinet import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def mil_prediction(pred, n=1):
    # generalized, returns top n 
    i = tf.argsort(pred[:, 1], axis=0)
    i = i[len(pred) - n : len(pred)]

    return (tf.gather(pred, i), i)

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    ########## HYPERPARAMETER SETUP ##########

    N_EPOCHS = 10000
    BATCH_SIZE = 2**6
    BUFFER_SIZE = BATCH_SIZE * 2
    ds = 8
    instance_size = (256, 256)
    volume_size = (256, 256, 160)
    num_classes = 6
    learning_rate = 1e-4
    progbar_length = 10
    CONVERGENCE_EPOCH_LIMIT = 10
    epsilon = 1e-4

    ########## DIRECTORY SETUP ##########

    MODEL_NAME = "phinet"
    WEIGHT_DIR = Path("models/weights") / MODEL_NAME
    RESULTS_DIR = Path("results")
    DATA_DIR = Path("Z:/data")

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

    print(model.summary(line_length=75))


    ######### FIVE FOLD CROSS VALIDATION #########

    TRAIN_CURVE_FILENAME = RESULTS_DIR / "training_curve_fold_{}.csv"

    TRAIN_TF_RECORD_FILENAME = DATA_DIR / "tfrecord_dir" / "dataset_fold_{}_train.tfrecord"
    VAL_TF_RECORD_FILENAME = DATA_DIR / "tfrecord_dir" / "dataset_fold_{}_val.tfrecord"
    
    for cur_fold in range(5):

        with open(str(TRAIN_CURVE_FILENAME).format(cur_fold), 'w') as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        ######### MODEL AND CALLBACKS #########
        model.load_weights(str(INIT_WEIGHT_PATH))
        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        ######### DATA IMPORT #########

        train_dataset = tf.data.TFRecordDataset(
                str(TRAIN_TF_RECORD_FILENAME).format(cur_fold))\
            .map(lambda record: parse_into_slice(
                record,
                instance_size,
                num_labels=num_classes))\
            .shuffle(BUFFER_SIZE)\
            .batch(BATCH_SIZE)

        val_dataset = tf.data.TFRecordDataset(
                str(VAL_TF_RECORD_FILENAME).format(cur_fold))\
            .map(lambda record: parse_into_volume(
                record,
                instance_size,
                num_labels=num_classes))

            
        '''
        augmentations = [flip_dim1, flip_dim2, rotate_2D]
        for f in augmentations:
            train_dataset = train_dataset.map(
                    lambda x, y: 
                    tf.cond(tf.random.uniform([], 0, 1) > 0.9, 
                        lambda: (f(x), y),
                        lambda: (x, y)
                    ), num_parallel_calls=4,)
        '''

        num_elements = 1920 * (256 + 256 + 160) #1920 volumes * all slices per volume

        # metrics
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
        val_accuracy = tf.keras.metrics.Accuracy(name='val_acc')
        val_loss = tf.keras.metrics.Mean(name='val_loss')

        # step
        def train_step(inputs):
            x, y = inputs
            
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = tf.nn.compute_average_loss(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=y,
                        logits=logits,
                    ),
                    global_batch_size=BATCH_SIZE,
                )

            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            train_accuracy.update_state(y, tf.nn.softmax(logits))
            return loss

        def val_step(inputs):
            x, y = inputs

            logits = model(x, training=False)

            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=y,
                logits=logits,
            )

            mean_loss = tf.reduce_mean(losses)
            val_loss.update_state(mean_loss)

            # arithmetic mean of prediction of all slices
            pred = tf.reduce_mean(
                tf.nn.softmax(logits),
                axis=0,
            )
            # update val acc by taking highest class: argmax
            val_accuracy.update_state(tf.argmax(y), tf.argmax(pred))


        ######### TRAINING #########

        train_loss = 0

        best_val_loss = 100000
        best_val_acc = 0
        convergence_epoch_counter = 0

        print()

        TEMPLATE = "\rEpoch {}/{} [{:{}<{}}] Loss: {:>3.4f} Acc: {:>3.2%}"

        sys.stdout.write(TEMPLATE.format(
            1, 
            N_EPOCHS, 
            "=" * 0, 
            '-', 
            progbar_length,
            0.0,
            0.0,
        ))

        best_epoch = 1
        for cur_epoch in range(N_EPOCHS):

            num_batches = 0
            print("\nTraining...")
            for i, data in enumerate(train_dataset):
                train_loss += train_step(data)
                num_batches += 1

                cur_step = BATCH_SIZE * (i + 1)

                sys.stdout.write(TEMPLATE.format(
                    cur_epoch + 1, N_EPOCHS,
                    "=" * min(int(progbar_length*(cur_step/num_elements)), 
                              progbar_length),
                    "-",
                    progbar_length,
                    train_loss/num_batches,
                    train_accuracy.result(),
                ))
                sys.stdout.flush()

            train_loss /= num_batches

            # validation metrics
            print("\nValidating...")
            num_val_elements = 1920 * 0.2 #(80/20 split of dev set)
            for i, (x, y) in enumerate(val_dataset):
                # avoid failures in TFRecord where no slices were gathered
                if x.shape[0] == 0:
                    continue

                val_step((x, y))

                cur_step = (i + 1)

                sys.stdout.write(TEMPLATE.format(
                    cur_epoch + 1, N_EPOCHS,
                    "=" * min(int(progbar_length*(cur_step/num_val_elements)), 
                              progbar_length),
                    "-",
                    progbar_length,
                    val_loss.result(),
                    val_accuracy.result(),
                ))
                sys.stdout.flush()
            

            with open(str(TRAIN_CURVE_FILENAME).format(cur_fold), 'a') as f:
                f.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    cur_epoch + 1,
                    train_loss,
                    train_accuracy.result(),
                    val_loss.result(),
                    val_accuracy.result(),
                ))


            if convergence_epoch_counter >= CONVERGENCE_EPOCH_LIMIT:
                print("\nCurrent Fold: {}\
                        \nNo improvement in {} epochs, model is converged.\
                        \nModel achieved best val loss at epoch {}.\
                        \nTrain Loss: {:.4f} Train Acc: {:.2%}\
                        \nVal   Loss: {:.4f} Val   Acc: {:.2%}".format(
                    cur_fold,
                    CONVERGENCE_EPOCH_LIMIT,
                    best_epoch,
                    train_loss, 
                    train_accuracy.result(),
                    val_loss.result(), 
                    val_accuracy.result(),
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
