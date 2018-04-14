'''
Samuel Remedios
NIH CC CNRM
Data processing script
'''

import os
import csv
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
from sklearn.utils import shuffle

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'


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
        parser.add_argument('--weightdir', required=True, action='store', dest='OUT_DIR',
                            help='Output directory where the trained models are written')
        parser.add_argument('--numcores', required=True, action='store', dest='numcores',
                            default='1', type=int,
                            help='Number of cores to preprocess in parallel with')
    elif session == "test":
        parser.add_argument('--infile', required=True, action='store', dest='INFILE',
                            help='Image to classify')
        parser.add_argument('--model', required=True, action='store', dest='model',
                            help='Model Architecture (.json) file')
        parser.add_argument('--weights', required=True, action='store', dest='weights',
                            help='Learnt weights (.hdf5) file')
        parser.add_argument('--result_dst', required=True, action='store', dest='OUTFILE',
                            help='Output filename (e.g. result.csv) to where the results are written')
    elif session == "validate":
        parser.add_argument('--datadir', required=True, action='store', dest='VAL_DIR',
                            help='Where the initial unprocessed data is')
        parser.add_argument('--model', required=True, action='store', dest='model',
                            help='Model Architecture (.json) file')
        parser.add_argument('--weights', required=True, action='store', dest='weights',
                            help='Learnt weights (.hdf5) file')
        parser.add_argument('--result_dst', required=True, action='store', dest='OUT_DIR',
                            help='Output directory where the results are written')
        parser.add_argument('--result_file', required=True, action='store', dest='OUTFILE',
                            help='Output directory where the results are written')
        parser.add_argument('--numcores', required=True, action='store', dest='numcores',
                            default='1', type=int,
                            help='Number of cores to preprocess in parallel with')
    else:
        print("Invalid session. Must be one of \"train\", \"validate\", or \"test\"")
        sys.exit()

    parser.add_argument('--classes', required=True, action='store', dest='classes',
                        help='Comma separated list of all classes, CASE-SENSITIVE')
    parser.add_argument('--gpuid', required=False, action='store', type=int, dest='GPUID',
                        help='For a multi-GPU system, the trainng can be run on different GPUs.'
                        'Use a GPU id (single number), eg: 1 or 2 to run on that particular GPU.'
                        '0 indicates first GPU.  Optional argument. Default is the first GPU.')
    parser.add_argument('--delete_preprocessed_dir', required=False, action='store', dest='clear',
                        default='n', help='delete all temporary directories. Enter either y or n. Default is n.')

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
    call = "mri_convert -odt uchar --crop 0 0 0 -c" + " " + filename + \
        " " + os.path.join(tmpdir, basename)
    if verbose == 0:
        call = call + " " + ">/dev/null"
    os.system(call)

    # reorient to RAI. Not necessary
    # call = reorient_script_path + " " + \
    #    os.path.join(tmpdir, basename) + " " + "RAI"
    infile = os.path.join(tmpdir, basename)
    outfile = os.path.join(tmpdir, "reorient_" + basename)
    call = "3dresample -orient RAI -inset " + infile + " -prefix " + outfile
    if verbose == 0:
        call = call + " " + ">/dev/null"
    os.system(call)

    # robustfov to make sure neck isn't included
    infile = os.path.join(tmpdir, "reorient_" + basename)
    outfile = os.path.join(tmpdir, "robust_" + basename)
    call = robustfov_script_path + " " + infile + " " +\
        outfile + " " + "160"
    if verbose == 0:
        call = call + " " + ">/dev/null"
    os.system(call)

    # 3dWarp to make images AC-PC aligned. Not necessary.  Ideally images should be
    # rigid registered to some template for uniformity, but rigid registration is slow
    # This is a faster way.  -newgrid 2 will resample the image to 2mm^3 resolution
    infile = os.path.join(tmpdir, "robust_" + basename)
    outfile = os.path.join(outdir, basename)
    call = "3dWarp -deoblique -NN -newgrid 2 -prefix" + " " + outfile + " " + infile
    if verbose == 0:
        call = call + " " + ">/dev/null 2>&1"
    os.system(call)

    # since the intensities are already [0,255], change the file from float to uchar to save space
    call = "fslmaths " + outfile + " " + outfile + " -odt char"
    os.system(call)

    # delete temporary files to save space, otherwise the temp directory takes more than 100GB
    call = "rm -f " + os.path.join(tmpdir, basename)
    os.system(call)
    call = "rm -f " + os.path.join(tmpdir, "robust_" + basename)
    os.system(call)

    all_filenames = os.listdir(outdir)
    for f in all_filenames:
        if os.path.basename(f) == basename:
            new_name = f

    return new_name


def preprocess_dir(train_dir, preprocess_dir, reorient_script_path, robustfov_script_path, classes, ncores):
    '''
    Preprocesses all files in train_dir into preprocess_dir using prepreocess.sh

    Params:
        - train_dir: string, path to where all the training images are kept
        - preprocess_dir: string, path to where all preprocessed images will be saved
        - reorient_script_path: string, path to bash script to reorient image
        - robustfov_script_path: string, path to bash script to robustfov image
    '''
    TMPDIR = os.path.join(
        preprocess_dir, "tmp_intermediate_preprocessing_steps")

    class_directories = [os.path.join(train_dir, x)
                         for x in os.listdir(train_dir)]
    class_directories.sort()

    print(classes)
    num_classes = len(classes)

    # preprocess all images
    print("*** PREPROCESSING ***")
    for class_dir in tqdm(class_directories):

        if not os.path.basename(class_dir) in classes:
            print("{} not in specified {}; omitting.".format(
                os.path.basename(class_dir),
                classes))
            continue

        if not os.path.exists(TMPDIR):
            os.makedirs(TMPDIR)
        preprocess_class_dir = os.path.join(
            preprocess_dir, os.path.basename(class_dir))

        if not os.path.exists(preprocess_class_dir):
            os.makedirs(preprocess_class_dir)

        if len(os.listdir(class_dir)) <= len(os.listdir(preprocess_class_dir)):
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

        # remove the intermediate preprocessing steps at every iteration, otherwise
        # disk usage goes beyond 100GB, with lots of training data
        shutil.rmtree(TMPDIR)
    # If the preprocessed data already exists, delete tmp_intermediate_preprocessing_steps
    if os.path.exists(TMPDIR):
        shutil.rmtree(TMPDIR)


def load_image(filename):
    img = nib.load(filename).get_data()
    img = np.reshape(img, (1,)+img.shape+(1,))
    MAX_VAL = 255  # consistent maximum intensity in preprocessing

    # linear scaling so all intensities are in [0,1]
    return np.divide(img, MAX_VAL)


def get_classes(classes):
    '''
    Params:
        - classes: list of strings
    Returns:
        - class_encodings: dictionary mapping an integer to a class_string
    '''
    class_list = classes
    class_list.sort()

    class_encodings = {x: class_list[x] for x in range(len(class_list))}

    return class_encodings


def load_data(data_dir, classes=None):
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
        - num_classes: integer, number of classes
        - img_shape: ndarray, shape of an individual image
    '''

    labels = []

    #################### CLASSIFICATION OF UNKNOWN DATA ####################

    if classes is None:
        all_filenames = []
        filenames = [x for x in os.listdir(data_dir)
                     if not os.path.isdir(os.path.join(data_dir, x))]
        filenames.sort()

        for f in tqdm(filenames):
            img = nib.load(os.path.join(data_dir, f)).get_data()
            img = np.reshape(img, img.shape+(1,))
            data.append(img)
            all_filenames.append(f)

        data = np.array(data)

        return data, all_filenames

    #################### TRAINING OR VALIDATION ####################

    # determine number of classes
    class_directories = [os.path.join(data_dir, x)
                         for x in os.listdir(data_dir)]
    class_directories.sort()

    print(classes)
    num_classes = len(classes)

    # set up all_filenames and class_labels to speed up shuffling
    all_filenames = []
    class_labels = {}
    i = 0
    for class_directory in class_directories:

        if not os.path.basename(class_directory) in classes:
            print("{} not in {}; omitting.".format(
                os.path.basename(class_directory),
                classes))
            continue

        class_labels[os.path.basename(class_directory)] = i
        i += 1
        for filename in os.listdir(class_directory):
            filepath = os.path.join(class_directory, filename)
            all_filenames.append(filepath)

    img_shape = nib.load(all_filenames[0]).get_data().shape
    data = np.empty(shape=((len(all_filenames),) +
                           img_shape + (1,)), dtype=np.uint8)

    # shuffle data
    all_filenames = shuffle(all_filenames, random_state=0)

    data_idx = 0  # pointer to index in data

    for f in tqdm(all_filenames):
        img = nib.load(f).get_data()
        img = np.asarray(img, dtype=np.uint8)

        # place this image in its spot in the data array
        data[data_idx] = np.reshape(img, (1,)+img.shape+(1,))
        data_idx += 1

        cur_label = f.split(os.sep)[-2]
        labels.append(to_categorical(
            class_labels[cur_label], num_classes=num_classes))

    labels = np.array(labels, dtype=np.uint8)
    print(data.shape)
    print(labels.shape)
    return data, labels, all_filenames, num_classes, data[0].shape


def record_results(csv_filename, args):

    filename, ground_truth, prediction, confidences = args

    if ground_truth is not None:
        fieldnames = [
            "filename",
            "ground_truth",
            "prediction",
            "confidences",
        ]


        # write to file the two sums
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w') as csvfile:
                fieldnames = fieldnames

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(csv_filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                "filename": filename,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidences": confidences,
            })
    else:
        fieldnames = [
            "filename",
            "prediction",
            "confidences",
        ]


        # write to file the two sums
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w') as csvfile:
                fieldnames = fieldnames

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(csv_filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                "filename": filename,
                "prediction": prediction,
                "confidences": confidences,
            })



def now():
    '''
    Returns a string format of current time, for use in checkpoint filenaming
    '''
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
