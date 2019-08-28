import numpy as np
import os
import sys
import pandas as pd

import tensorflow as tf
from tensorflow.python_io import TFRecordWriter
import nibabel as nib

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm
from utils.pad import *
from utils import preprocess
from utils.tfrecord_utils import *
from utils.patch_ops import *


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing cmd line argument")
        sys.exit()

    tf.enable_eager_execution()

    ######### DIRECTRY SETUP #########

    # pass the preprocessed data directory here
    DATA_DIR = sys.argv[1]

    class_dirs = os.listdir(DATA_DIR)
    
    TF_RECORD_FILENAME = os.path.join(
            "data", "tfrecord_dir", "dataset_fold_{}_{}.tfrecord"
    )

    TARGET_DIMS = (256, 256)

    ######### GET DATA FILENAMES #######
    classes = os.listdir(DATA_DIR)
    class_mapping = {c:i for c, i in zip(classes, range(len(classes)))}
    class_mapping_inv = {v:k for k, v in class_mapping.items()}

    X_names = []
    y = []
    class_counter = {c:0 for c in classes}

    for root, _, files in os.walk(DATA_DIR):
        if len(files) > 0:
            for f in files:
                X_names.append(os.path.join(root, f))
                cur_class = root.split(os.sep)[-1]
                y.append(class_mapping[cur_class])
                class_counter[cur_class] += 1

    print("Initial class distribution:")
    for c, count in class_counter.items():
        print("{}: {}".format(c, count))

    X_names = np.array(X_names)
    y = np.array(y)

    class_indices = [np.where(y==i)[0] for i in range(len(classes))]
    class_indices = [shuffle(c, random_state=4) for c in class_indices]

    ######### TRAIN/TEST SPLIT #########
    LIMIT_TRAIN_SPLIT = int(0.8 * min([len(i) for i in class_indices]))
    print("\nTraining distribution:")
    for i, n in enumerate(class_indices):
        print("Number of train samples for class {}: {}".format(
                class_mapping_inv[i],
                len(n[:LIMIT_TRAIN_SPLIT])
            )
        )
    print("\nTesting distribution:")
    for i, n in enumerate(class_indices):
        print("Number of test samples for class {}: {}".format(
                class_mapping_inv[i],
                len(n[LIMIT_TRAIN_SPLIT:])
            )
        )

    train_idx = np.concatenate([
        c[:LIMIT_TRAIN_SPLIT] for c in class_indices
    ])

    test_idx = np.concatenate([
        c[LIMIT_TRAIN_SPLIT:] for c in class_indices
    ])

    # shuffle indices for randomness
    train_idx = shuffle(train_idx, random_state=4)
    test_idx = shuffle(test_idx, random_state=4)

    # split
    X_names_train = X_names[train_idx]
    y_train = y[train_idx]

    X_names_test = X_names[test_idx]
    y_test = y[test_idx]

    ######### 5-FOLD TRAIN/VAL SPLIT #########
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

    def prepare_data(x_filename, y_label, num_classes):
        x = nib.load(x_filename).get_fdata()
        x_slices = get_slices(x, TARGET_DIMS)
        x_slices = x_slices.astype(np.uint8)

        y = np.zeros((num_classes,), dtype=np.uint8)
        y[y_label] = 1

        return x_slices, y


    # K Fold for train and val
    for i, (train_idx, val_idx) in enumerate(skf.split(X_names_train, y_train)):

        print("Number training samples: {}\nNumber val samples: {}".format(
            len(train_idx),
            len(val_idx),
        ))

        for cur_idx, cur_name in [(train_idx, "train"), (val_idx, "val")]:
            print("\nCreating {} TFRecord...".format(cur_name))
            with TFRecordWriter(TF_RECORD_FILENAME.format(i, cur_name)) as writer:
                x_slices = []
                y_labels = []
                counter = 0
                SHUFFLE_LIMIT = 300

                for num, cur_i in tqdm(enumerate(cur_idx), total=len(cur_idx)):
                    cur_x_slices, cur_y_label = prepare_data(
                        X_names_train[cur_i],
                        y_train[cur_i],
                        len(classes),
                    )

                    for cur_x_slice in cur_x_slices:
                        x_slices.append(cur_x_slice)
                        y_labels.append(cur_y_label)

                    counter += 1

                    if cur_name == "train":
                        # shuffle all the slices after loading some or remaining chunk
                        if counter >= SHUFFLE_LIMIT or num == len(cur_idx)-1:
                            x_slices, y_labels = shuffle(x_slices, y_labels)
                            for x_slice, y_label in zip(x_slices, y_labels):
                                tf_example = slice_image_example(x_slice, y_label)
                                writer.write(tf_example.SerializeToString())
                            x_slices = []
                            y_labels = []
                            counter = 0


                    # Val set is written as volumes
                    else:
                        tf_example = volume_image_example(
                            cur_x_slices, 
                            cur_y_label, 
                            len(cur_x_slices)
                        )
                        writer.write(tf_example.SerializeToString())



    # Testing images are written as full volumes!
    with TFRecordWriter(TF_RECORD_FILENAME.format("_", "test")) as writer:
        for x_name, y_label in tqdm(zip(X_names_test, y_test), total=len(X_names_test)):
            x_slices, y_label = prepare_data(
                x_name,
                y_label,
                len(classes),
            )

            tf_example = volume_image_example(x_slices, y_label, len(x_slices))
            writer.write(tf_example.SerializeToString())
