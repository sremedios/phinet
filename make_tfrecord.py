import os
import sys
from pathlib import Path
from itertools import repeat

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python_io import TFRecordWriter
import nibabel as nib

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

from utils.pad import *
from utils import preprocess
from utils.tfrecord_utils import *
from utils.patch_ops import *


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing cmd line argument")
        sys.exit()

    tf.enable_eager_execution()

    ######### DIRECTORY SETUP #########

    # pass the preprocessed data directory here
    DATA_DIR = Path(sys.argv[1])

    TF_RECORD_FILENAME = os.path.join(
            "data", "tfrecord_dir", "dataset_fold_{}_{}.tfrecord"
    )
    TEST_FILENAMES_FILE = os.path.join(
            "data", "test_filenames_labels.txt"
    )

    TARGET_DIMS = (256, 256)

    ######### GET DATA FILENAMES #######
    classes = sorted(DATA_DIR.iterdir())
    class_mapping = {c.name:i for i, c in enumerate(sorted(DATA_DIR.iterdir()))}
    class_mapping_inv = {v:k for k, v in class_mapping.items()}

    X_names = []
    y = []
    class_counter = {c.name:0 for c in classes}

    for classdir in DATA_DIR.iterdir():
        for filename in classdir.iterdir():
            X_names.append(filename)
            cur_class = filename.parts[-2]
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
        x = nib.load(str(x_filename)).get_fdata()
        x_slices = get_slices(x, TARGET_DIMS)

        y = np.zeros((num_classes,), dtype=np.uint8)
        y[y_label] = 1

        return x_slices, y

    # write which filenames and classes for train/val/test
    train_filenames_file = Path("data/train_filenames_fold_{}.txt")
    val_filenames_file = Path("data/val_filenames_fold_{}.txt")
    test_filenames_file = Path("data/test_filenames_fold_{}.txt")

    def yield_tf_example(
            cur_idx,
            X_names_train,
            y_train,
            classes,
            train_filenames_file,
            val_filenames_file,
    ):
        x_slices = []
        y_labels = []
        counter = 0
        for num, cur_i in tqdm(enumerate(cur_idx), total=len(cur_idx)):
            cur_x_slices, cur_y_label = prepare_data(
                X_names_train[cur_i],
                y_train[cur_i],
                len(classes),
            )

            x_slices.extend(cur_x_slices)
            y_labels.extend(repeat(cur_y_label, len(cur_x_slices)))

            counter += 1

            if cur_name == "train":
                # shuffle all the slices after loading some or remaining chunk
                with open(str(train_filenames_file).format(i), 'a') as f:
                    f.write("{},{}\n".format(
                        X_names_train[cur_i],
                        y_train[cur_i],
                    ))

                if counter >= SHUFFLE_LIMIT or num == len(cur_idx)-1:
                    x_slices, y_labels = shuffle(x_slices, y_labels)
                    for x_slice, y_label in zip(x_slices, y_labels):
                        tf_example = slice_image_example(x_slice, y_label)
                        yield tf_example

                    x_slices = []
                    y_labels = []
                    counter = 0


            # Val set is written as volumes
            else:
                with open(str(val_filenames_file).format(i), 'a') as f:
                    f.write("{},{}\n".format(
                        X_names_train[cur_i],
                        y_train[cur_i],
                    ))
                tf_example = volume_image_example(
                    cur_x_slices, 
                    cur_y_label, 
                    len(cur_x_slices)
                )
                yield tf_example



    # K Fold for train and val
    for i, (train_idx, val_idx) in enumerate(skf.split(X_names_train, y_train)):

        print("Number training samples: {}\nNumber val samples: {}".format(
            len(train_idx),
            len(val_idx),
        ))

        for cur_idx, cur_name in [(train_idx, "train"), (val_idx, "val")]:
            print("\nCreating {} TFRecord...".format(cur_name))
            with TFRecordWriter(TF_RECORD_FILENAME.format(i, cur_name)) as writer:
                # total num slices * classes * 2
                SHUFFLE_LIMIT = 256 * 3 * len(classes) * 2 

                for tf_example in yield_tf_example(
                    cur_idx,
                    X_names_train,
                    y_train,
                    classes,
                    train_filenames_file,
                    val_filenames_file,
                ):
                    writer.write(tf_example.SerializeToString())



    '''
    # Testing images are NOT written

    # Testing occurs directly on the raw nifti volumes
    # to validate the preprocessing steps




    # Testing images are written as full volumes!
    with TFRecordWriter(TF_RECORD_FILENAME.format("_", "test")) as writer:
        for x_name, y_label in tqdm(zip(X_names_test, y_test), total=len(X_names_test)):
            with open(TEST_FILENAMES_FILE, 'a') as f:
                f.write("{},{}\n".format(x_name, y_label))
            x_slices, y_label = prepare_data(
                x_name,
                y_label,
                len(classes),
            )

            tf_example = volume_image_example(x_slices, y_label, len(x_slices))
            writer.write(tf_example.SerializeToString())
    '''
