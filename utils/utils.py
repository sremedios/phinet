'''
Samuel Remedios
NIH CC CNRM
Data processing script
'''

import os
import random
from tqdm import *
import argparse
import glob
import shutil
from joblib import Parallel, delayed
import numpy as np
import nibabel as nib
import sys
from datetime import datetime
from keras.utils import to_categorical


def parse_args(session):
    '''
    Parse command line arguments.

    Params:
        - session: string, one of "train", "validate", or "test"
    Returns:
        - parse_args: object, accessible representation of args
    '''
    parser = argparse.ArgumentParser(
        description="Arguments for Training and Testing")

    if session == "train":
        parser.add_argument('--datadir', required=True, action='store', dest='TRAIN_DIR',
                            help='Where the initial unprocessed data is')
        parser.add_argument('--o', required=True, action='store', dest='OUT_DIR',
                            help='Output directory where the trained models are written')
    elif session == "test":
        parser.add_argument('--infile', required=True, action='store', dest='INFILE',
                            help='Image to classify')
        parser.add_argument('--model', required=True, action='store', dest='model',
                            help='Learnt model (.hdf5) file')
        parser.add_argument('--o', required=True, action='store', dest='OUTFILE',
                            help='Output filepath and name to where the results are written')
        parser.add_argument('--preprocesseddir', required=True, action='store',
                            dest='PREPROCESSED_DIR',
                            help='Output directory where final preprocessed images are placed ')
    elif session == "validate":
        parser.add_argument('--datadir', required=True, action='store', dest='VAL_DIR',
                            help='Where the initial unprocessed data is')
        parser.add_argument('--model', required=True, action='store', dest='model',
                            help='Learnt model (.hdf5) file')
        parser.add_argument('--o', required=True, action='store', dest='OUTFILE',
                            help='Output directory where the results are written')
    else:
        print("Invalid session. Must be one of \"train\", \"validate\", or \"test\"")
        sys.exit()

    parser.add_argument('--numcores', required=False, action='store', dest='numcores',
                        default='1', type=int,
                        help='Number of cores to preprocess in parallel with')
    parser.add_argument('--task', required=True, action='store', dest='task',
                        help='Type of task: modality, T1-contrast, FL-contrast')
    parser.add_argument('--gpuid', required=False, action='store', type=int, dest='GPUID',
                        help='For a multi-GPU system, the trainng can be run on different GPUs.'
                        'Use a GPU id (single number), eg: 0 or 1 to run on that particular GPU.'
                        '0 indicates first GPU.  Optional argument. Default is the last GPU.')
    parser.add_argument('--delete_preprocessed_dir', required=False, action='store', dest='clear',
                        default='n', help='delete tmp directory')

    return parser.parse_args()


def preprocess(filename, outdir, tmpdir, reorient_script_path, robustfov_script_path, verbose=1):
    '''
    Preprocess a single file.
    Can be used in parallel

    Params:
        - filename: string, path to file to preprocess
        - outdir: string, path to destination directory to save preprocessed image
        - tmpdir: string, path to tmp directory for intermediate steps
        - reorient_script_path: string, path to bash script to reorient image
        - robustfov_script_path: string, path to bash script to robustfov image
        - verbose: int, if 0, surpress all output. If 1, display all output

    Returns:
        - string, name of new file in its new location
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    basename = os.path.basename(filename)

    # convert image to 256^3 1mm^3 coronal images with intensity range [0,255]
    call = "mri_convert -c" + " " + filename + \
        " " + os.path.join(tmpdir, basename)
    if verbose == 0:
        call = call + " " + ">/dev/null"
    os.system(call)

    # reorient to RAI. Not necessary
    call = reorient_script_path + " " + \
        os.path.join(tmpdir, basename) + " " + "RAI"
    if verbose == 0:
        call = call + " " + ">/dev/null"
    os.system(call)

    # robustfov to make sure neck isn't included
    call = robustfov_script_path + " " + os.path.join(tmpdir, basename) + " " +\
        os.path.join(tmpdir, "robust_" + basename) + " " + "160"
    if verbose == 0:
        call = call + " " + ">/dev/null"
    os.system(call)

    # 3dWarp to make images AC-PC aligned. Not necessary.  Ideally images should be
    # rigid registered to some template for uniformity, but rigid registration is slow
    # This is a faster way.  -newgrid 2 will resample the image to 2mm^3 resolution
    outfile = os.path.join(outdir, basename)
    infile = os.path.join(tmpdir, "robust_" + basename)
    call = "3dWarp -deoblique -NN -newgrid 2 -prefix" + " " + outfile + " " + infile
    if verbose == 0:
        call = call + " " + ">/dev/null"
    os.system(call)

    return os.listdir(outdir)[0]


def preprocess_dir(train_dir, preprocess_dir, reorient_script_path, robustfov_script_path, ncores):
    '''
    Preprocesses all files in train_dir into preprocess_dir using prepreocess.sh

    Params:
        - train_dir: string, path to where all the training images are kept
        - preprocess_dir: string, path to where all preprocessed images will be saved
        - reorient_script_path: string, path to bash script to reorient image
        - robustfov_script_path: string, path to bash script to robustfov image
    '''
    TMPDIR = os.path.join(preprocess_dir, "tmp_intermediate_preprocessing_steps")
    if not os.path.exists(TMPDIR):
        os.makedirs(TMPDIR)

    class_directories = [os.path.join(train_dir, x)
                         for x in os.listdir(train_dir)]
    class_directories.sort()
    num_classes = len(class_directories)

    # preprocess all images
    print("*** PREPROCESSING ***")
    for class_dir in tqdm(class_directories):
        preprocess_class_dir = os.path.join(
            preprocess_dir, os.path.basename(class_dir))

        if not os.path.exists(preprocess_class_dir):
            os.makedirs(preprocess_class_dir)

        if len(os.listdir(class_dir)) == len(os.listdir(preprocess_class_dir)):
            print("Already preprocessed.")
            continue

        filenames = [os.path.join(class_dir, x)
                     for x in os.listdir(class_dir)]

        # preprocess in parallel using all but one cores (n_jobs=-2)
        Parallel(n_jobs=ncores)(delayed(preprocess)(filename=f,
                                                outdir=preprocess_class_dir,
                                                tmpdir=TMPDIR,
                                                reorient_script_path=reorient_script_path,
                                                robustfov_script_path=robustfov_script_path,
                                                verbose=0,)
                            for f in filenames)

    # remove the intermediate preprocessing steps
    shutil.rmtree(TMPDIR)


def load_image(filename):
    img = [nib.load(filename).get_data()]
    img = np.array(img)
    return img


def get_classes(task):
    class_encodings = {}

    if task=="modality":
        class_encodings = {0: "FL",
                           1: "T1",
                           2: "T2",}
    elif task=="t1-contrast":
        class_encodings = {0: "T1 Post",
                           1: "T1 Pre",}
    elif task=="fl-contrast":
        class_encodings = {0: "FL Post",
                           1: "FL Pre",}
    else:
        print("Invalid task: must be one of \"modality\", \"t1-contrast\", \"fl-contrast\"")
        sys.exit()
        
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
