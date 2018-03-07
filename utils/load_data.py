'''
Samuel Remedios
NIH CC CNRM
Data processing script
'''

import os
from tqdm import *
import numpy as np
import nibabel as nib
import sys
from keras.utils import to_categorical


def load_data_binary_class(root_dir, test=False, mode="FL", augment=False,
                           eval_confs=False, retrain=False, model_name=None,
                           part=1, LIMIT=100000):
    '''
    Loads in datasets and returns tensor data and labels for
    the MR images.
    Params:
        - directory: where the data is
    Returns:
        - data: a list of all tensors
        - masks: a list of all mask tensors
        - labels: a list of all labels
        - affines: a list of all affines
        - headers: a list of all headers
        - filenames: a list of all corresponding filenames
    '''

    if mode == "T1":
        ALL_DIRS = ["T1", "T1C"]
    else:
        ALL_DIRS = ["FL", "FLC"]

    all_filenames = []
    data = []
    # hard coding shape for now
    # data = np.empty([128,128,128],dtype=np.float16)
    labels = []
    for directory in ALL_DIRS:
        print("Processing directory: " + directory + "...")

        if not test:
            CUR_DIR = os.path.join(root_dir, "fully_processed", directory)
        else:
            CUR_DIR = os.path.join(root_dir, "fully_processed_test", directory)
        filenames = os.listdir(CUR_DIR)
        filenames.sort()
        # limit examples for RAM for whatever reason
        if part == 1:
            filenames = filenames[:LIMIT]
        else:
            filenames = filenames[LIMIT:]

        # remove all confident filenames
        if retrain:
            unconf_path = os.path.join(root_dir, "predictions", model_name)
            with open(os.path.join(unconf_path, "unconfident_filenames"), 'r') as f:
                unconfs = f.read()
            unconfs = unconfs.splitlines()
            filenames = list(set(filenames).intersection(set(unconfs)))

        if test or eval_confs:
            # keep track of all filenames for testing
            for f in filenames:
                all_filenames.append(f)

        print("Loading objects...")
        count = 0
        total = len(filenames)
        for f in tqdm(filenames):
            data.append(nib.load(os.path.join(CUR_DIR, f)).get_data())
            if len(data[-1].shape) != 3:
                data.remove(data[-1])
            if directory == "T1" or directory == "FL":
                labels.append(0)
            else:
                labels.append(1)

            if augment:
                x, y, z = flip(data[-1])
                for item in [x, y, z]:
                    data.append(item)
                    if directory == "T1" or directory == "FL":
                        labels.append(0)
                    else:
                        labels.append(1)


    sys.stdout.write("Data processing complete.\n")

    print("Converting to nparray")
    data = np.asarray(data, dtype=np.float16)

    labels = np.asarray(labels)
    labels = np.reshape(labels, labels.shape + (1,))
    if test or eval_confs:
        return data, labels, all_filenames
    return data, labels


def load_data_multiclass(root_dir, test=False, augment=False, eval_confs=False,
                         retrain=False, model_name=None):
    '''
    Loads in datasets and returns tensor data and labels for
    the MR images.
    Params:
        - root_dir: where the data is
        - test: boolean, if this function is being used for test or train
        - augment: boolean, applies flips over axes if true
        - eval_confs: (deprecated, previously used for retraining over less confident predictions)
        - retrain: (deprecated)
        - model_name: one of "
    Returns:
        - data: a list of all tensors
        - labels: a list of all labels
        - filenames: a list of all corresponding filenames
    '''

    ALL_DIRS = os.listdir(root_dir)

    all_filenames = []
    # hard coding shape for now: must align with what images were padded to
    #data = np.empty((1,128,128,128),dtype=np.float16)
    data = []
    labels = []
    for directory in ALL_DIRS:
        print("Processing directory: " + directory + "...")

        if not test:
            CUR_DIR = os.path.join(root_dir, directory)
        else:
            #CUR_DIR = os.path.join(root_dir, "test", directory)
            CUR_DIR = os.path.join(root_dir, directory)
        filenames = os.listdir(CUR_DIR)
        filenames.sort()

        if test or eval_confs:
            # keep track of all filenames for testing
            for f in filenames:
                all_filenames.append(f)

        print("Loading objects...")
        count = 0
        total = len(filenames)
        for f in tqdm(filenames):
            data.append(nib.load(os.path.join(CUR_DIR, f)).get_data())

            # set up labels
            if directory == "T1":
                labels.append(0)
            elif directory == "T2":
                labels.append(1)
            else:
                labels.append(2)

            # data augmentation
            if augment:
                x, y, z = flip(data[-1])
                for item in [x, y, z]:
                    data.append(item)
                    if directory == "T1":
                        labels.append(0)
                    elif directory == "T2":
                        labels.append(1)
                    else:
                        labels.append(2)


    sys.stdout.write("Data processing complete.\n")

    print("Converting to ndarray")
    data = np.asarray(data, dtype=np.float16)

    labels = to_categorical(labels)
    if test or eval_confs:
        return data, labels, all_filenames
    return data, labels


def load_data(filename, FILE_DIR):
    '''
    Loads in datasets and returns tensor data and labels for
    the MR images.
    '''
    data = [nib.load(os.path.join(FILE_DIR, filename)).get_data()]
    data = np.asarray(data, dtype=np.float16)
    return data


def preprocess_training_data(src_dir, dst_root, target_res=3, target_shape=(96, 96, 112)):
    '''
    Preprocess all files for all modalities in training and validation data.
    These preprocessed files will be deleted at the end of the file. 
    '''

    # downsample all training data into same directory heirarchy
    modalities = os.listdir(src_dir)
    modality_paths = [os.path.join(src_dir, x) for x in os.listdir(src_dir)
                      if x != "predictions"]
    for modality, modality_path in zip(modalities, modality_paths):
        filenames = os.listdir(modality_path)
        for filename in filenames:
            dst_dir = os.path.join(dst_root, modality)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            downsample(filename, modality_path, dst_dir, target_res)

            # normalize and pad each file individually
            img_obj = nib.load(os.path.join(dst_dir, filename))
            img_data, affine, header = extract_elements(img_obj)
            img_data = pad(
                img_data, target_shape[0], target_shape[1], target_shape[2])
            img_data = normalize_data(img_data)
            nii_obj = nib.Nifti1Image(img_data, affine=affine, header=header)
            nib.save(nii_obj, os.path.join(dst_dir, filename))


def downsample(filename, FILE_DIR, TARGET_DIR, target_size):
    size_argument = " ".join([str(x) for x in [target_size]*3])
    os.system("3dresample -dxyz " + size_argument + " -inset " +
              os.path.join(FILE_DIR, filename) + " -prefix " +
              os.path.join(TARGET_DIR, filename) + " -rmode Li")


def normalize_data(img, contrast='T1'):
    '''
    Normalizes 3D images via KDE and clamping
    Params:
        - img: 3D image
    Returns:
        - normalized image
    '''
    from statsmodels.nonparametric.kde import KDEUnivariate
    from scipy.signal import argrelextrema

    if contrast == 'T1':
        CONTRAST = 1
    else:
        CONTRAST = 0

    if (len(np.nonzero(img)[0])) == 0:
        normalized_img = img
    else:
        tmp = np.asarray(np.nonzero(img.flatten()))
        q = np.percentile(tmp, 99.)
        tmp = tmp[tmp <= q]
        tmp = np.asarray(tmp, dtype=float).reshape(-1, 1)

        GRID_SIZE = 80
        bw = float(q) / GRID_SIZE

        kde = KDEUnivariate(tmp)
        kde.fit(kernel='gau', bw=bw, gridsize=GRID_SIZE, fft=True)
        X = 100.*kde.density
        Y = kde.support

        idx = argrelextrema(X, np.greater)
        idx = np.asarray(idx, dtype=int)
        H = X[idx]
        H = H[0]
        p = Y[idx]
        p = p[0]
        x = 0.

        if CONTRAST == 1:
            T1_CLAMP_VALUE = 1.25
            x = p[-1]
            normalized_img = img/x
            normalized_img[normalized_img > T1_CLAMP_VALUE] = T1_CLAMP_VALUE
        else:
            T2_CLAMP_VALUE = 3.5
            x = np.amax(H)
            j = np.where(H == x)
            x = p[j]
            if len(x) > 1:
                x = x[0]
            normalized_img = img/x
            normalized_img[normalized_img > T2_CLAMP_VALUE] = T2_CLAMP_VALUE

    normalized_img /= normalized_img.max()
    return normalized_img


def pad(image, x_max, y_max, z_max):
    '''
    Zero-pads all three dimensions of the image up to x_max/2, y_max/2, and z_max/2.

    Params:
        - image: ndarray (3d) of image data
        - x/y/z_max: the amount to pad data to
    Returns:
        - the padded image data
    '''

    a, b, c = image.shape
    # calc difference between current and max, accounting for all-sides padding
    x, y, z = np.abs(a - x_max)//2, np.abs(b - y_max)//2, np.abs(c - z_max)//2

    # pad
    new_img = np.pad(image, ((x, x), (y, y), (z, z)),
                     mode="constant", constant_values=0)

    # adjust for odd indices in original image by padding additional row
    if a % 2 != 0:
        new_img = np.pad(new_img, ((0, 1), (0, 0), (0, 0)),
                         mode="constant", constant_values=0)
    if b % 2 != 0:
        new_img = np.pad(new_img, ((0, 0), (0, 1), (0, 0)),
                         mode="constant", constant_values=0)
    if c % 2 != 0:
        new_img = np.pad(new_img, ((0, 0), (0, 0), (0, 1)),
                         mode="constant", constant_values=0)

    return new_img


def extract_elements(nii_obj):
    '''
    Separates the nifti objects into their constituent parts
    Params:
        - nii_objs: a list of nifti objects
    Returns:
        - data: list of ndarrays of image data
        - affines: list of ndarrays of affine data
        - headers: list of nibabel headers
    '''
    data = nii_obj.get_data()
    affines = nii_obj.affine
    headers = nii_obj.header
    return data, affines, headers
