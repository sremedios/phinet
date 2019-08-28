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

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    #opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.98)
    #conf = tf.compat.v1.ConfigProto(gpu_options=opts)
    #tf.compat.v1.enable_eager_execution(config=conf)
    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    #tf.compat.v1.enable_eager_execution()
    ########## DISTRIBUTION STRATEGY ##########
    strategy = tf.distribute.MirroredStrategy()

    ########## HYPERPARAMETER SETUP ##########

    N_EPOCHS = 10000
    BATCH_SIZE_PER_REPLICA = 4096 
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    BUFFER_SIZE = GLOBAL_BATCH_SIZE * 2
    ds = 2
    instance_size = (256, 256)
    volume_size = (256, 256, 160)
    num_classes = 6
    learning_rate = 1e-4
    progbar_length = 10
    CONVERGENCE_EPOCH_LIMIT = 10
    epsilon = 1e-4

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

    with strategy.scope():
        # Actual instantiation happens for each fold
        model = phinet(num_classes=num_classes, ds=ds)
        #model = resnet(num_classes=num_classes, ds=ds)
    
    INIT_WEIGHT_PATH = os.path.join(WEIGHT_DIR, "init_weights.h5")
    model.save_weights(INIT_WEIGHT_PATH)
    json_string = model.to_json()
    with open(MODEL_PATH, 'w') as f:
        json.dump(json_string, f)

    print(model.summary(line_length=75))


    ######### FIVE FOLD CROSS VALIDATION #########

    TRAIN_CURVE_FILENAME = os.path.join(
            RESULTS_DIR, "training_curve_fold_{}.csv"
    )

    TRAIN_TF_RECORD_FILENAME = os.path.join(
            "data", "tfrecord_dir", "dataset_fold_{}_train.tfrecord"
    )
    VAL_TF_RECORD_FILENAME = os.path.join(
            "data", "tfrecord_dir", "dataset_fold_{}_val.tfrecord"
    )

    for cur_fold in range(5):

        with open(TRAIN_CURVE_FILENAME.format(cur_fold), 'w') as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        ######### MODEL AND CALLBACKS #########
        with strategy.scope():
            model.load_weights(INIT_WEIGHT_PATH)
            opt = tf.optimizers.Adam(learning_rate=learning_rate)

        ######### DATA IMPORT #########
        augmentations = [flip_dim1, flip_dim2, rotate_2D]

        train_dataset = tf.data.TFRecordDataset(
                TRAIN_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_into_slice(
                record,
                instance_size,
                num_labels=num_classes))\
            .shuffle(BUFFER_SIZE)\
            .batch(BATCH_SIZE_PER_REPLICA)\

        val_dataset = tf.data.TFRecordDataset(
                VAL_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_into_volume(
                record,
                instance_size,
                num_labels=num_classes))\

        '''
        for f in augmentations:
            train_dataset = train_dataset.map(
                    lambda x, y: 
                    tf.cond(tf.random.uniform([], 0, 1) > 0.9, 
                        lambda: (f(x), y),
                        lambda: (x, y)
                    ), num_parallel_calls=4,)
        '''

        num_elements = 1920 * 160 #1920 volumes * 160 slices per volume

        ######### DISTRIBUTE SETUP #########

        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

        with strategy.scope():
            # metrics
            train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
            val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_acc')
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
                            #reduction=tf.losses.Reduction.NONE,
                        ),
                        global_batch_size=GLOBAL_BATCH_SIZE,
                    )

                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))

                train_accuracy.update_state(y, tf.nn.softmax(logits))
                return loss

            def val_step(inputs):
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

                val_accuracy.update_state(y, pred)
                val_loss.update_state(tf.reduce_sum(losses))


            @tf.function
            def distributed_train_step(dataset_inputs):
                per_replica_losses = strategy.experimental_run_v2(
                        train_step,
                        args=(dataset_inputs,)
                )
                return strategy.reduce(
                        tf.distribute.ReduceOp.SUM,
                        per_replica_losses,
                        axis=None,
                )

            @tf.function
            def distributed_val_step(dataset_inputs):
                return strategy.experimental_run_v2(
                        val_step,
                        args=(dataset_inputs,)
                )
    
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
        with strategy.scope():
            for cur_epoch in range(N_EPOCHS):

                num_batches = 0
                for i, data in enumerate(train_dist_dataset):
                    train_loss += distributed_train_step(data)
                    num_batches += 1

                    cur_step = BATCH_SIZE_PER_REPLICA * (i + 1)

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
                num_val_elements = 0
                for i, data in enumerate(val_dataset):
                    distributed_val_step(data)

                with open(TRAIN_CURVE_FILENAME.format(cur_fold), 'a') as f:
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
                    model.save_weights(os.path.join(
                        WEIGHT_DIR, "best_weights_fold_{}.h5".format(cur_fold))
                    )

                sys.stdout.write(" Val Loss: {:.4f} Val Acc: {:.2%}".format(
                    val_loss.result(),
                    val_accuracy.result(), 
                ))

