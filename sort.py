'''
Samuel Remedios
NIH CC CNRM
Auto sorts data using phinet.
'''

import os
from time import time
from operator import itemgetter
from shutil import move
import nibabel as nib
import numpy as np

start_time = time()
from models.phinet import phinet
print("Elapsed time to load tensorflow:", time()-start_time)
from utils.load_data import load_data, downsample, normalize_data, extract_elements, pad
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = ""

start_time = time()
DATA_DIR = "sorting_example"
TMP_DIR = os.path.join(DATA_DIR, "tmp")
if not os.path.exists(TMP_DIR): os.makedirs(TMP_DIR)

UNSORTED_DIR = os.path.join(DATA_DIR, "unsorted")
filenames = os.listdir(UNSORTED_DIR)
for filename in filenames:
    # first preprocessing step
    # downsample
    print("Downsampling...")
    downsample(filename, UNSORTED_DIR, TMP_DIR, 3)

    # normalize and pad
    print("Normalizing and padding...")
    img_obj = nib.load(os.path.join(TMP_DIR, filename))
    img_data, affine, header = extract_elements(img_obj)
    img_data = pad(img_data, 96,96,112)
    img_data = normalize_data(img_data)
    nii_obj = nib.Nifti1Image(img_data, affine=affine, header=header)
    nib.save(nii_obj, os.path.join(TMP_DIR, filename))


    # load data
    print("Loading processed data...")
    X = load_data(filename, TMP_DIR)
    INPUT_SHAPE = X[0].shape
    weights = os.path.join("weights", "phinet_contrast_2.hdf5")

    print("Loading model...")
    model_contrast_class = phinet(INPUT_SHAPE, n_inputs=3, load_weights=True, weights=weights)
    print("Predicting...")
    preds = model_contrast_class.predict(X, batch_size=1, verbose=1)
    max_idx, _ = max(enumerate(preds[0]), key=itemgetter(1))
    if max_idx == 0: contrast = "T1"
    elif max_idx == 1: contrast = "T2"
    else: contrast = "FL"

    print("{} is a: {}".format(filename, contrast))

    TARGET_DIR = os.path.join(DATA_DIR, contrast)

    if contrast != 'T2':
        # secondary preprocessing step due to current training of phinet

        # must remove file for downsample code to overwrite
        os.remove(os.path.join(TMP_DIR, filename))

        # downsample
        downsample(filename, UNSORTED_DIR, TMP_DIR, 2)

        # normalize and pad
        img_obj = nib.load(os.path.join(TMP_DIR, filename))
        img_data, affine, header = extract_elements(img_obj)
        img_data = pad(img_data, 128,128,128)
        img_data = normalize_data(img_data, contrast=contrast)
        nii_obj = nib.Nifti1Image(img_data, affine=affine, header=header)
        nib.save(nii_obj, os.path.join(TMP_DIR, filename))
    
        # load appropriate weights
        if contrast == 'T1':
            weights = os.path.join("weights", "phinet_T1.hdf5")
        elif contrast == 'FL':
            weights = os.path.join("weights", "phinet_FL.hdf5")

        # run pre v post if T1 or FLAIR
        X = load_data(filename, TMP_DIR)
        INPUT_SHAPE = X[0].shape
        model_pre_v_post = phinet(INPUT_SHAPE, n_inputs=1, load_weights=True, weights=weights)
        preds = model_pre_v_post.predict(X, batch_size=1, verbose=0)
        if preds[0] < 0.5: 
            TARGET_DIR = os.path.join(TARGET_DIR, "pre")
        else: 
            TARGET_DIR = os.path.join(TARGET_DIR, "post")

    print("Moving data...")
    # move file
    if not os.path.exists(TARGET_DIR): os.makedirs(TARGET_DIR)
    move(os.path.join(UNSORTED_DIR, filename), os.path.join(TARGET_DIR, filename))

    # empty tmp directory
    os.remove(os.path.join(TMP_DIR, filename))

# delete tmp directory
time.sleep(0.1)
os.rmdir(TMP_DIR)

print("Elapsed time:", time() - start_time)
K.clear_session()
