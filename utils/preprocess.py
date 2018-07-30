'''
Samuel Remedios
NIH CC CNRM
Preprocess files
'''

import os
import subprocess
from multiproecessing.pool import ThreadPool
from tqdm import *
import shutil
from joblib import Parallel, delayed
import sys

from .mri_convert import mri_convert
from .robustfov import robust_fov

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'


def preprocess(filename, preprocess_dir, verbose=1, remove_tmp_files=True):
    '''
    Preprocess a single file.
    Can be used in parallel

    Params:
        - filename: string, path to file to preprocess
        - preprocess_dir: string, path to destination directory to save preprocessed image
        - verbose: int, if 0, surpress all output. If 1, display all output
    '''
    MRI_CONVERT_DIR = os.path.join(preprocess_dir, "mri_convert")
    REORIENT_DIR = os.path.join(preprocess_dir, "reorient")
    ROBUST_FOV_DIR = os.path.join(preprocess_dir, "robust_fov")
    WARP_3D_DIR = os.path.join(preprocess_dir, "warp3d")

    for d in [MRI_CONVERT_DIR, REORIENT_DIR, ROBUST_FOV_DIR, WARP_3D_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    mri_convert(filename, src_dir, MRI_CONVERT_DIR)
    reorient(filename, MRI_CONVERT_DIR, REORIENT_DIR)
    robust_fov(filename, REORIENT_DIR, ROBUST_FOV_DIR)
    warp_3d(filename, ROBUST_FOV_DIR, WARP_3D_DIR)

    # since the intensities are already [0,255] after warp3d,
    # change the file from float to uchar to save space
    call = "fslmaths " + outfile + " " + outfile + " -odt char"
    subprocess.call(call)

    # move final preprocess step into the preprocessing directory
    shutil.move(os.path.join(filename, WARP_3D_DIR),
                os.path.join(filename, preprocess_dir))

    # remove the intermediate steps from each of the preprocessing steps
    if remove_tmp_files:
        for d in [MRI_CONVERT_DIR, REORIENT_DIR, ROBUST_FOV_DIR, WARP_3D_DIR]:
            tmp_file = os.path.join(d, filename)
            os.remove(tmp_file)


# TODO: refactor this completely
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
        tp = ThreadPool(30)
        for f in filenames:
            tp.apply_async(preprocess(f,
                                      preprocess_dir,
                                      verbose=0,
                                      remove_tmp_files=True))
        tp.close()
        tp.join()
        """
        Parallel(n_jobs=ncores)(delayed(preprocess)(filename=f,
                                                    outdir=preprocess_class_dir,
                                                    tmpdir=TMPDIR,
                                                    reorient_script_path=reorient_script_path,
                                                    robustfov_script_path=robustfov_script_path,
                                                    verbose=0,)
                                for f in filenames)
        """

        # remove the intermediate preprocessing steps at every iteration, otherwise
        # disk usage goes beyond 100GB, with lots of training data
        shutil.rmtree(TMPDIR)
    # If the preprocessed data already exists, delete tmp_intermediate_preprocessing_steps
    if os.path.exists(TMPDIR):
        shutil.rmtree(TMPDIR)
