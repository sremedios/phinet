'''
Samuel Remedios
NIH CC CNRM
Data processing script
'''

import os
import random
from tqdm import *
import numpy as np
import nibabel as nib
import sys
from keras.utils import to_categorical


def robust_fov(data_dir, dst_dir):
    '''
    Calls fsl's robustfov on all images in the given directory, outputting them 
    into a directory at the same level called "robustfov"

    Params:
        - data_dir: string, path to data from which to remove necks
        - preprocess_dir: string, path to where the data will be saved
    '''
    filenames = [x for x in os.listdir(data_dir) 
            if not os.path.isdir(os.path.join(data_dir,x))]

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for filename in filenames:
        filepath = os.path.join(data_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.exists(dst_path):
            continue
        else:
            call = "robustfov -i " + filepath + " -r " + dst_path + " >/dev/null"
            os.system(call)


def load_data(data_dir, preprocess_dir, patch_size, labels_known=True):
    '''
    Loads in datasets and returns the labeled preprocessed patches for use in the model.

    Determines the number of classes for the problem and assigns labels to each class,
    sorted alphabetically.

    Params:
        - data_dir: string, path to all training class directories
        - preprocess_dir: string, path to destination for robustfov files
        - patch_size: 3-element tuple of integers, size of patches to use for training
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
        print("*** CALLING ROBUSTFOV ***")
        robust_fov(data_dir, preprocess_dir)

        filenames = [x for x in os.listdir(preprocess_dir) 
                if not os.path.isdir(os.path.join(preprocess_dir,x))]
        filenames.sort()

        for f in filenames:
            img = nib.load(os.path.join(preprocess_dir, f)).get_data()
            normalized_img = normalize_data(img)
            patches = get_patches(normalized_img, patch_size)

            for patch in tqdm(patches):
                data.append(patch)
                all_filenames.append(f)

        print("A total of {} patches collected.".format(len(data)))

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
    class_encodings_file = os.path.join(data_dir, "..", "..", "class_encodings.txt")
    if not os.path.exists(class_encodings_file):
        with open(class_encodings_file, 'w') as f:
            for i in range(len(class_directories)):
                f.write(os.path.basename(class_directories[i]) + " " + str(i) + '\n')

    # robustfov all images
    print("*** CALLING ROBUSTFOV ***")
    for class_dir in tqdm(class_directories):
        dst_dir = os.path.join(preprocess_dir, os.path.basename(class_dir))
        robust_fov(class_dir, dst_dir)

    # point to the newly-processed files
    class_directories = [os.path.join(preprocess_dir, x)
                         for x in os.listdir(preprocess_dir)]
    class_directories.sort()

    print("*** GATHERING PATCHES ***")
    for i in range(len(class_directories)):
        filenames = os.listdir(class_directories[i])
        filenames.sort()

        for f in filenames:
            img = nib.load(os.path.join(class_directories[i], f)).get_data()
            normalized_img = normalize_data(img)
            patches = get_patches(normalized_img, patch_size)

            for patch in tqdm(patches):
                data.append(patch)
                labels.append(to_categorical(i, num_classes=num_classes))
                all_filenames.append(f)

    print("A total of {} patches collected.".format(len(data)))

    data = np.array(data, dtype=np.float16)
    labels = np.array(labels, dtype=np.float16)

    return data, labels, all_filenames


def get_patches(img, patch_size, num_patches=1000):
    '''
    Gets num_patches 3D patches of the input image for classification.

    Patches may overlap.

    The center of each patch is some random distance from the center of
    the entire image, where the random distance is drawn from a Gaussian dist.

    Params:
        - img: 3D ndarray, the image data from which to get patches
        - patch_size: 3-element tuple of integers, size of the 3D patch to get
        - num_patches: integer (default=100), number of patches to retrieve
    Returns:
        - patches: list of 3D ndarrays, the resultant 3D patches
    '''
    # set random seed and variable params
    random.seed()
    mu = 0
    sigma = 30

    # find center of the given image
    center_coords = [x//2 for x in img.shape]

    # find num_patches random numbers as distances from the center
    patches = []
    for _ in range(num_patches):
        horizontal_displacement = int(random.gauss(mu, sigma))
        depth_displacement = int(random.gauss(mu, sigma))
        # deviate half as much vertically
        vertical_displacement = int(random.gauss(mu, sigma//2))

        # current center coords
        c = [
            center_coords[0] + horizontal_displacement,
            center_coords[1] + depth_displacement,
            center_coords[2] + vertical_displacement
        ]

        if c[0]+patch_size[0]//2 > img.shape[0] or c[0]-patch_size[0]//2 < 0 or\
            c[1]+patch_size[1]//2 > img.shape[1] or c[1]-patch_size[1]//2 < 0 or\
                c[2]+patch_size[2]//2 > img.shape[2] or c[2]-patch_size[2]//2 < 0:
            continue

        # get patch
        patch = img[
            c[0]-patch_size[0]//2:c[0]+patch_size[0]//2+1,
            c[1]-patch_size[1]//2:c[1]+patch_size[1]//2+1,
            c[2]-patch_size[2]//2:c[2]+patch_size[2]//2+1,
        ]

        if patch.shape != patch_size:
            continue

        patches.append(patch)

    return patches


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
