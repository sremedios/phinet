import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
import nibabel as nib

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

from utils.pad import *
from utils import preprocess
from utils.tfrecord_utils import *
from utils.patch_ops import *

def prepare_data(x_filename, y_label, target_dims, num_classes):
    x = nib.load(str(x_filename)).get_fdata(dtype=np.float32)
    x_slices = get_axial_slices(x, target_dims)

    y = np.array(tf.one_hot(y_label, depth=num_classes), dtype=np.float32)

    return x_slices, y


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing cmd line argument")
        sys.exit()

    ######### DIRECTORY SETUP #########

    # pass the preprocessed data directory here
    DATA_DIR = Path(sys.argv[1])
    target_fold = int(sys.argv[2])

    TFRECORD_DIR = Path("data/tfrecord_dir")
    if not TFRECORD_DIR.exists():
        TFRECORD_DIR.mkdir(parents=True)
    TFRECORD_FNAME = TFRECORD_DIR / "dataset_fold_{}_{}.tfrecord"

    TARGET_DIMS = (256, 256)

    ######### GET DATA FILENAMES #######
    classes = sorted(DATA_DIR.iterdir())
    class_to_int = {c.name:i for i, c in enumerate(classes)}
    int_to_class = {v:k for k, v in class_to_int.items()}

    # get all filenames
    fnames = sorted(
           set([fname.resolve() for classdir in DATA_DIR.iterdir()
                for fname in classdir.iterdir()]),
           key=lambda f: f.parent.name,
        )

    # group by class, (folder)
    fnames_by_class = {k:list(g) for k, g in\
            itertools.groupby(fnames, lambda f: f.parent.name)}

    print("Initial class distribution:")
    for c, fname_list in fnames_by_class.items():
        print("{}: {}".format(c, len(fname_list)))

    X_names = np.array(fnames)
    y = np.array(
            list(map(lambda fname:\
                class_to_int[fname.parent.name],
                fnames,
            )),
            dtype=np.float32,
        )

    class_indices = [np.where(y==i)[0] for i in range(len(classes))]
    class_indices = [shuffle(c, random_state=4) for c in class_indices]

    ######### TRAIN/TEST SPLIT #########
    LIMIT_TRAIN_SPLIT = int(0.8 * min([len(i) for i in class_indices]))
    print("\nTraining distribution:")
    for i, n in enumerate(class_indices):
        print("Number of train samples for class {}: {}".format(
                int_to_class[i],
                len(n[:LIMIT_TRAIN_SPLIT])
            )
        )
    print("\nTesting distribution:")
    for i, n in enumerate(class_indices):
        print("Number of test samples for class {}: {}".format(
                int_to_class[i],
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

    # write which filenames and classes for train/val/test
    train_filenames_file = Path("data/train_filenames_fold_{}.txt")
    val_filenames_file = Path("data/val_filenames_fold_{}.txt")
    test_filenames_file = Path("data/test_filenames.txt")
    data_count_file = Path("data/data_count.txt")

    def yield_tf_example(
            cur_idx,
            X_names_train,
            y_train,
            classes,
            train_filenames_file,
            val_filenames_file,
    ):
        # total num slices * classes * 10
        SHUFFLE_LIMIT = 256 * len(classes) * 10
        x_slices = []
        y_labels = []
        counter = 0
        for num, cur_i in tqdm(enumerate(cur_idx), total=len(cur_idx)):

            cur_x_slices, cur_y_label = prepare_data(
                X_names_train[cur_i],
                y_train[cur_i],
                TARGET_DIMS,
                len(classes),
            )

            x_slices.extend(cur_x_slices)
            y_labels.extend(itertools.repeat(cur_y_label, len(cur_x_slices)))

            counter += 1

            if cur_name == "train":
                # shuffle all the slices after loading some or remaining chunk
                with open(str(train_filenames_file).format(i), 'a') as f:
                    f.write("{},{}\n".format(
                        X_names_train[cur_i],
                        int(y_train[cur_i]),
                    ))

                if counter >= SHUFFLE_LIMIT or num == len(cur_idx)-1:
                    x_slices, y_labels = shuffle(x_slices, y_labels, random_state=counter)
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
                        int(y_train[cur_i]),
                    ))
                tf_example = volume_image_example(
                    cur_x_slices, 
                    cur_y_label, 
                    len(cur_x_slices)
                )
                yield tf_example



    # K Fold for train and val
    for cur_fold, (train_idx, val_idx) in enumerate(skf.split(X_names_train, y_train)):

        if cur_fold != target_fold:
            continue

        print("Number training samples: {}\nNumber val samples: {}".format(
            len(train_idx),
            len(val_idx),
        ))


        for cur_idx, cur_name in [(train_idx, "train"), (val_idx, "val")]:
            print("\nCreating {} TFRecord...".format(cur_name))

            instance_counter = 0
            
            for tf_example in yield_tf_example(
                cur_idx,
                X_names_train,
                y_train,
                classes,
                str(train_filenames_file).format(target_fold),
                str(val_filenames_file).format(target_fold),
            ):
                instance_counter += 1

            '''
            with tf.io.TFRecordWriter(str(TFRECORD_FNAME).format(cur_fold, cur_name)) as writer:

                for tf_example in yield_tf_example(
                    cur_idx,
                    X_names_train,
                    y_train,
                    classes,
                    str(train_filenames_file).format(target_fold),
                    str(val_filenames_file).format(target_fold),
                ):
                    instance_counter += 1
                    writer.write(tf_example.SerializeToString())

            with open(data_count_file, 'a') as f:
                f.write("{},{},{}\n".format(cur_name, cur_fold, instance_counter))
            '''


    # Write just the test fnames to file
    for x_name, y_label in tqdm(zip(X_names_test, y_test), total=len(X_names_test)):
        with open(test_filenames_file, 'a') as f:
            f.write("{},{}\n".format(x_name, y_label))
        #with open(data_count_file, 'a') as f:
            #f.write("{},{},{}\n".format("test", "_", len(test_idx)))


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
