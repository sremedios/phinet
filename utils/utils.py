'''
Samuel Remedios
NIH CC CNRM
Data processing script
'''

import os
import random
from tqdm import *
import argparse
import numpy as np
import nibabel as nib
import sys
from datetime import datetime
from keras.utils import to_categorical


def parse_training_args():
    parser = argparse.ArgumentParser(
        description="Arguments for Training and Testing")

    parser.add_argument('--task', required=True, action='store', dest='task',
                        help='Type of task: modality, T1-contrast, FL-contrast')
    parser.add_argument('--traindir', required=True, action='store', dest='TRAIN_DIR',
                        help='Where the initial unprocessed data is')
    parser.add_argument('--gpuid', required=False, action='store', type=int, dest='GPUID',
                        help='For a multi-GPU system, the trainng can be run on different GPUs.'
                        'Use a GPU id (single number), e.g. 0 or 1 to run on that particular GPU.'
                        '0 indicates first GPU.  Optional argument. Default is the last GPU.')
    parser.add_argument('--o', required=True, action='store', dest='OUT_DIR',
                        help='Output directory where the trained models are written')

    return parser.parse_args()

def parse_testing_args():
    parser = argparse.ArgumentParser(
        description="Arguments for Training and Testing")

    parser.add_argument('--infile', required=True, action='store', dest='INFILE',
                        help='Image to classify')
    parser.add_argument('--gpuid', required=False, action='store', type=int, dest='GPUID',
                        help='For a multi-GPU system, the trainng can be run on different GPUs.'
                        'Use a GPU id (single number), e.g. 0 or 1 to run on that particular GPU.'
                        '0 indicates first GPU.  Optional argument. Default is the last GPU.')
    parser.add_argument('--encodings', required=True, action='store', dest='encodings_file',
                        help='File holding encodings')
    parser.add_argument('--model', required=True, action='store', dest='model',
                        help='Learnt model (.hdf5) file')
    parser.add_argument('--delete_preprocessed_dir', required=False, action='store', dest='clear',
                        default='n', help='delete tmp directory')
    parser.add_argument('--o', required=True, action='store', dest='OUTFILE',
                        help='Output directory where the results are written')
    parser.add_argument('--outdir', required=True, action='store', dest='OUT_DIR',
                        help='Output directory where the preprocessing files are written')

    return parser.parse_args()

def preprocess_file(infile, dst_dir, script_path):
    '''
    Preprocesses a single file
    '''
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    call = "sh" + " " + script_path + " " + infile + " " + dst_dir 
    os.system(call)
    return os.listdir(dst_dir)[0]

def preprocess_dir(train_dir, preprocess_dir, script_path):
    '''
    Preprocesses all files in train_dir into preprocess_dir using prepreocess.sh

    Params:
        - train_dir: string, path to where all the training images are kept
        - preprocess_dir: string, path to where all preprocessed images will be saved
        - script_path: string, path to the preprocess script
    '''

    class_directories = [os.path.join(train_dir, x)
                         for x in os.listdir(train_dir)]
    class_directories.sort()
    num_classes = len(class_directories)

    # preprocess all images
    print("*** PREPROCESSING ***")
    for class_dir in tqdm(class_directories):
        preprocess_class_dir = os.path.join(
            preprocess_dir, os.path.basename(class_dir))

        if os.path.exists(preprocess_class_dir) and \
                len(os.listdir(class_dir)) == len(os.listdir(preprocess_class_dir)):
            print("Already preprocessed.")
            continue

        if not os.path.exists(preprocess_class_dir):
            os.makedirs(preprocess_class_dir)

        filenames = [os.path.join(class_dir, x)
                     for x in os.listdir(class_dir)]

        for filename in tqdm(filenames):
            call = "sh" + " " + script_path + " " + filename + " " + preprocess_class_dir
            os.system(call)

def load_image(filename):
    img = [nib.load(filename).get_data()]
    img = np.array(img)
    return img

def get_classes(encoding_file):
    class_encodings = {}
    with open(encoding_file, 'r') as f:
        content = f.read().split('\n')
    for line in content:
        if len(line)==0:
            continue
        entry = line.split()
        class_encodings[int(entry[1])] = entry[0]
    return class_encodings

def load_data(data_dir, labels_known=True):
    '''
    Loads in datasets and returns the labeled preprocessed patches for use in the model.

    Determines the number of classes for the problem and assigns labels to each class,
    sorted alphabetically.

    Params:
        - data_dir: string, path to all training class directories
        - task: string, one of modality, T1-contrast, FL-contrast'
        - labels_known: boolean, True if we know the labels, such as for training or
                                 validation.  False if we do not know the labels, such
                                 as loading in data to classify in production
    Returns:
        - data: list of 3D ndarrays, the patches of images to use for training
        - labels: list of 1D ndarrays, one-hot encoding corresponding to classes
        - all_filenames: list of strings, corresponding filenames for use in validation/test
    '''

    data = []
    labels = []
    all_filenames = []

    #################### CLASSIFICATION OF UNKNOWN DATA ####################

    if not labels_known:
        filenames = [x for x in os.listdir(data_dir)
                     if not os.path.isdir(os.path.join(data_dir, x))]
        filenames.sort()

        for f in filenames:
            img = nib.load(os.path.join(data_dir, f)).get_data()
            data.append(img)
            all_filenames.append(f)

        data = np.array(data)

        return data, all_filenames

    #################### TRAINING OR VALIDATION ####################

    # determine number of classes
    class_directories = [os.path.join(data_dir, x)
                         for x in os.listdir(data_dir)]
    class_directories.sort()
    num_classes = len(class_directories)

    # write the mapping of class to a local file in the following space-separated format:
    # CLASS_NAME integer_category
    class_encodings_file = os.path.join(
        data_dir, "..","..", "..", "class_encodings.txt")
    if not os.path.exists(class_encodings_file):
        with open(class_encodings_file, 'w') as f:
            for i in range(len(class_directories)):
                f.write(os.path.basename(
                    class_directories[i]) + " " + str(i) + '\n')

    # point to the newly-processed files
    class_directories = [os.path.join(data_dir, x)
                         for x in os.listdir(data_dir)]
    class_directories.sort()

    for i in range(len(class_directories)):
        filenames = os.listdir(class_directories[i])
        filenames.sort()

        for f in filenames:
            img = nib.load(os.path.join(class_directories[i], f)).get_data()
            data.append(img)
            labels.append(to_categorical(i, num_classes=num_classes))
            all_filenames.append(f)

    data = np.array(data, dtype=np.float16)
    labels = np.array(labels, dtype=np.float16)

    return data, labels, all_filenames


def now():
    '''
    Returns a string format of current time, for use in checkpoint filenaming
    '''
    return datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")
