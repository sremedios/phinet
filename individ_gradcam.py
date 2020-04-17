'''

NOTE: for now, images MUST be preprocessed ahead of time

IE: the fnames file provided must point to preprocessed images

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from pathlib import Path
import json
import numpy as np
from scipy.stats import mode
import nibabel as nib
import cv2
import tensorflow as tf
from tqdm import tqdm
from models.phinet import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def slice_gradcam(target_feature_map, pooled_grad, slice_shape):
    gradcam = tf.reduce_mean(
        target_feature_map * pooled_grad,
        axis=-1,
    )
    gradcam = np.array(tf.nn.relu(gradcam))
    if gradcam.max() != 0:
        #gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
        gradcam = cv2.resize(
            gradcam, 
            (slice_shape[1], slice_shape[0]), #cv2 expects opposite from numpy 
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        gradcam = np.zeros(slice_shape)
    return gradcam

def generate_gradcam(axial_slices, model, target_class_idx):
    # move axial slice idx back to idx 2
    out_vol = np.zeros(
            (
                axial_slices.shape[1],
                axial_slices.shape[2],
                axial_slices.shape[0],
            ),
            dtype=np.float32,
        )

    with tf.GradientTape() as tape:
        target_feature_maps, logits = model(axial_slices, training=False)
        logits = logits[:, target_class_idx]

    grads = tape.gradient(logits, target_feature_maps)
    if grads is None:
        print("None gradient returned")
        return out_vol
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    for slice_idx in range(len(axial_slices)):
        if axial_slices[slice_idx].sum() == 0:
            continue
        out_vol[:, :, slice_idx] = slice_gradcam(
                target_feature_maps[slice_idx],
                pooled_grads[slice_idx],
                out_vol.shape[0:2],
            )
    return out_vol


if __name__ == "__main__":

    WEIGHT_DIR = Path(sys.argv[1])
    fname = Path(sys.argv[2])
    GPUID = sys.argv[3]

    os.environ['CUDA_VISIBLE_DEVICES'] = GPUID

    ########## DIRECTORY SETUP ##########

    MODEL_NAME = WEIGHT_DIR.name
    TRAINED_WEIGHTS_FILENAME = WEIGHT_DIR / "best_weights_fold_{}.h5"
    LAYER_NAME = "concatenate"
    MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")
    RESULTS_DIR = Path("results") / MODEL_NAME
    DST_DIR = RESULTS_DIR / "figures" / "gradcams"

    classes = sorted(['FL','FLC','PD','T1','T1C','T2'])
    int_to_class = {i:c for i,c in enumerate(classes)}
    class_to_int = {v:k for k,v in int_to_class.items()}

    ######### INFERENCE #########

    true_class = Path(fname.parent).name
    individual_dir = DST_DIR / true_class / (fname.name.split('.')[0])
    if not individual_dir.exists():
        individual_dir.mkdir(parents=True)

    for cur_class in classes:

        dst_fname = individual_dir / "gradcam_target_{}_{}.nii.gz".format(
                cur_class,
                fname.name.split('.')[0],
            )


        obj = nib.load(fname)
        slice_shape = (obj.shape[0], obj.shape[2])
        axial_slices = np.moveaxis(obj.get_fdata(dtype=np.float32), 2, 0)[..., np.newaxis]
        gradcam_vol = np.zeros(nib.load(fname).shape, dtype=np.float32)

        fold_preds = []
        for cur_fold in range(5):
            ### LOAD MODEL ###
            with open(MODEL_PATH) as json_data:
                model = tf.keras.models.model_from_json(json.load(json_data))
            model.load_weights(str(TRAINED_WEIGHTS_FILENAME).format(cur_fold))
            model = tf.keras.models.Model(
                model.inputs,
                [model.get_layer(LAYER_NAME).output, model.output],
            )

            _, logits = model(axial_slices, training=False)
            preds = tf.nn.softmax(logits)
            pred = tf.reduce_mean(preds, axis=0)
            fold_preds.append(tf.argmax(pred).numpy())


            ### GEN GRADCAM ###
            gradcam_vol = running_average(
                    gradcam_vol,
                    generate_gradcam(axial_slices, model, class_to_int[cur_class]),
                    cur_fold + 1,
                )

        gradcam_nii_obj = nib.Nifti1Image(gradcam_vol, obj.affine, header=obj.header)

        nib.save(gradcam_nii_obj, dst_fname)

    print("Pred class: {}".format(
        int_to_class[int(mode(fold_preds)[0])]
    ))
