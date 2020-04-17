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

from utils.tfrecord_utils import *

def get_axial_slices(img_vol):
    # rearrange for axial idx to be first idx
    tmp = np.moveaxis(img_vol, 2, 0)
    # remove top and bottom 25 slices due to instability in preprocessing
    tmp = tmp[25:]
    tmp = tmp[:-25]
    # cast to numpy array and add channel dimension
    tmp = np.array(tmp)[..., np.newaxis]
    return tmp

def normalize(img):
    # clamp to 99th percentile, then [0,1] normalize
    q = np.percentile(img[np.nonzero(img)],99)
    img[img>q] = q
    img = (img - img.min()) / (img.max() - img.min())
    return img


def prepare_data(x_filename, y_label, num_classes):
    x = nib.load(str(x_filename)).get_fdata(dtype=np.float32)
    x = get_axial_slices(x)
    x = normalize(x)

    y = np.array(tf.one_hot(y_label, depth=num_classes), dtype=np.float32)

    return x, y


if __name__ == "__main__":

    ######### DIRECTORY SETUP #########

    # pass the preprocessed data directory here
    DATA_DIR = Path(sys.argv[1])
    TFRECORD_DIR = Path(sys.argv[2])
    #exclusion_list = Path(sys.argv[3])
    cur_fold = int(sys.argv[3])

    if not TFRECORD_DIR.exists():
        TFRECORD_DIR.mkdir(parents=True)
    TFRECORD_FNAME = TFRECORD_DIR / "dataset_fold_{}_{}.tfrecord"


    ######### GET DATA FILENAMES #######
    classes = sorted(['FL', 'FLC', 'PD', 'T1', 'T1C', 'T2'])
    class_to_int = {c:i for i, c in enumerate(classes)}
    int_to_class = {v:k for k, v in class_to_int.items()}

    # get all filenames
    fnames = sorted(
           set([fname.resolve() for classdir in DATA_DIR.iterdir()
                for fname in classdir.iterdir()]),
           key=lambda f: f.parent.name,
        )

    '''
    exclusions = list(map(lambda l: l.strip(), open(exclusion_list, 'r')))

    def check_exclusions(fname, exclusions):
        for e in exclusions:
            if fname.startswith(e):
                return True
        return False

    fnames = sorted(filter(lambda f: not check_exclusions(f.name, exclusions), fnames))
    '''

    # group by class, (folder)
    fnames_by_class = {k:list(g) for k, g in\
            itertools.groupby(fnames, lambda f: f.parent.name)}

    print("Initial class distribution:")
    for c, fname_list in fnames_by_class.items():
        print("{}: {}".format(c, len(fname_list)))

    fnames = np.array(fnames)
    fnames = shuffle(fnames, random_state=100)
    y = np.array(
            list(map(lambda fname: class_to_int[fname.parent.name], fnames,)),
            dtype=np.float32,
        )
    # get indices per class
    class_indices = [np.where(y==i)[0] for i in range(len(classes))]

    ######### DEV/TEST SPLIT #########
    LIMIT_DEV_SPLIT = int(0.8 * min([len(i) for i in class_indices]))
    print("\nTraining distribution:")
    for i, n in enumerate(class_indices):
        print("Number of dev samples for class {}: {}".format(
                int_to_class[i],
                len(n[:LIMIT_DEV_SPLIT])
            )
        )
    print("\nTesting distribution:")
    for i, n in enumerate(class_indices):
        print("Number of test samples for class {}: {}".format(
                int_to_class[i],
                len(n[LIMIT_DEV_SPLIT:])
            )
        )

    dev_idx = np.concatenate([c[:LIMIT_DEV_SPLIT] for c in class_indices])
    test_idx = np.concatenate([c[LIMIT_DEV_SPLIT:] for c in class_indices])

    # split
    fnames_dev = fnames[dev_idx].copy()
    y_dev = y[dev_idx].copy()

    fnames_test = fnames[test_idx].copy()
    y_test = y[test_idx].copy()

    ######### 5-FOLD TRAIN/VAL SPLIT #########

    # write which filenames and classes for train/val/test
    train_filenames_file = TFRECORD_DIR / "train_filenames_fold_{}.txt"
    val_filenames_file = TFRECORD_DIR / "val_filenames_fold_{}.txt"
    test_filenames_file = TFRECORD_DIR / "test_filenames.txt"
    data_count_file = TFRECORD_DIR / "data_count.txt"

    # shuffle before split to keep folds different
    fnames_dev, y_dev = shuffle(fnames_dev, y_dev, random_state=cur_fold)

    LIMIT_TRAIN_SPLIT = int(0.8 * len(fnames_dev))
    fnames_train = fnames_dev[:LIMIT_TRAIN_SPLIT]
    y_train = y_dev[:LIMIT_TRAIN_SPLIT]
    fnames_val = fnames_dev[LIMIT_TRAIN_SPLIT:]
    y_val = y_dev[LIMIT_TRAIN_SPLIT:]

    # shuffle after split to ensure mixed classes
    fnames_train, y_train = shuffle(fnames_train, y_train, random_state=cur_fold)
    fnames_val, y_val = shuffle(fnames_val, y_val, random_state=cur_fold)

    print("Current Fold: {}\nNumber training samples: {}\nNumber val samples: {}".format(
        cur_fold,
        len(fnames_train),
        len(y_train),
    ))

    ##### TRAIN TFRECORD #####
    print("Writing TRAIN TFRecord...")
    with tf.io.TFRecordWriter(str(TFRECORD_FNAME).format(cur_fold, "train")) as writer:
        x_slices = []
        y_labels = []
        instance_counter = 0
        SHUFFLE_LIMIT = 182 * 20 # 182 slices x 20 patients to mix up

        for i, (fname, y) in tqdm(enumerate(zip(fnames_train, y_train)), total=len(y_train)):
            cur_x_slices, cur_y_label = prepare_data(fname, y, len(classes))
            x_slices.extend(cur_x_slices)
            y_labels.extend(itertools.repeat(cur_y_label, len(cur_x_slices)))

            if len(y_labels) >= SHUFFLE_LIMIT or i >= len(y_train)-1:
                x_slices, y_labels = shuffle(x_slices, y_labels, random_state=0)
                for x_slice, y_label in zip(x_slices, y_labels):
                    tf_example = slice_image_example(x_slice, y_label)
                    writer.write(tf_example.SerializeToString())
                    instance_counter += 1
                x_slices = []
                y_labels = []

            with open(str(train_filenames_file).format(cur_fold), 'a') as f:
                f.write("{},{}\n".format(fname, int(y)))

    with open(data_count_file, 'a') as f:
        f.write("{},{},{}\n".format("train", cur_fold, instance_counter))

    ##### VAL TFRECORD #####
    print("Writing VAL TFRecord...")
    with tf.io.TFRecordWriter(str(TFRECORD_FNAME).format(cur_fold, "val")) as writer:
        instance_counter = 0

        for i, (fname, y) in tqdm(enumerate(zip(fnames_val, y_val)), total=len(y_val)):
            cur_x_slices, cur_y_label = prepare_data(fname, y, len(classes))
            tf_example = volume_image_example(cur_x_slices, cur_y_label, len(cur_x_slices))
            writer.write(tf_example.SerializeToString())
            instance_counter += 1

            with open(str(val_filenames_file).format(cur_fold), 'a') as f:
                f.write("{},{}\n".format(fname, int(y)))

    with open(data_count_file, 'a') as f:
        f.write("{},{},{}\n".format("val", cur_fold, instance_counter))


    ##### TEST FILENAMES #####
    if cur_fold == 0:
        print("Writing TEST filenames...")
        for fname, y in tqdm(zip(fnames_test, y_test), total=len(fnames_test)):
            with open(test_filenames_file, 'a') as f:
                f.write("{},{}\n".format(fname, int(y)))
