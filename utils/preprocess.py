'''
Samuel Remedios
NIH CC CNRM
Preprocess files
'''

import os
from tqdm import *
import shutil
from joblib import Parallel, delayed
import sys

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

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
        call = call + " " + ">/dev/null  2>&1"
    os.system(call)

    # reorient to RAI. Not necessary
    # call = reorient_script_path + " " + \
    #    os.path.join(tmpdir, basename) + " " + "RAI"
    infile = os.path.join(tmpdir, basename)
    outfile = os.path.join(tmpdir, "reorient_" + basename)
    call = "3dresample -orient RAI -inset " + infile + " -prefix " + outfile
    if verbose == 0:
        call = call + " " + ">/dev/null  2>&1 "
    os.system(call)

    # robustfov to make sure neck isn't included
    infile = os.path.join(tmpdir, "reorient_" + basename)
    outfile = os.path.join(tmpdir, "robust_" + basename)
    call = robustfov_script_path + " " + infile + " " +\
        outfile + " " + "160"
    if verbose == 0:
        call = call + " " + ">/dev/null  2>&1"
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
