'''
Samuel Remedios
NIH CC CNRM
Patch operations
'''

import os
import random
from tqdm import *
import numpy as np
import nibabel as nib
from keras.utils import to_categorical
from .reorient import orient
from .robustfov import robust_fov

def load_patch_data(data_dir, preprocess_dir, patch_size, labels_known=True, num_patches=100):
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
        print("*** CALLING 3DRESAMPLE ***")
        orient_dir = orient(data_dir, preprocess_dir)

        print("*** CALLING ROBUSTFOV ***")
        robustfov_dir = robust_fov(orient_dir, preprocess_dir)

        filenames = [x for x in os.listdir(robustfov_dir) 
                if not os.path.isdir(os.path.join(robustfov_dir,x))]
        filenames.sort()

        for f in filenames:
            img = nib.load(os.path.join(robustfov_dir, f)).get_data()
            #normalized_img = normalize_data(img)
            patches = get_patches(img, patch_size, num_patches)

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

    print("*** GATHERING PATCHES ***")
    for i in range(len(class_directories)):
        filenames = os.listdir(class_directories[i])
        filenames.sort()

        for f in tqdm(filenames):
            img = nib.load(os.path.join(class_directories[i], f)).get_data()
            #normalized_img = normalize_data(img)
            patches = get_patches(img, patch_size, num_patches)

            for patch in patches:
                data.append(patch)
                labels.append(to_categorical(i, num_classes=num_classes))
                all_filenames.append(f)

    print("A total of {} patches collected.".format(len(data)))

    data = np.array(data, dtype=np.float16)
    data = np.reshape(data, (data.shape + (1,))) 
    labels = np.array(labels, dtype=np.float16)

    return data, labels, all_filenames


def get_patches(img, patch_size, num_patches=100):
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
