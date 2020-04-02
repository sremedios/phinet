'''

NOTE: for now, images MUST be preprocessed ahead of time

IE: the filenames file provided must point to preprocessed images

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from pathlib import Path
import json

import numpy as np
import nibabel as nib
import cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tqdm import tqdm

from utils.pad import *
from utils import preprocess
from utils.patch_ops import get_axial_slices
from utils.augmentations import *
from utils.tfrecord_utils import *
from models.phinet import *

int_to_class = {i: c for i, c in 
        enumerate(sorted(['FL', 'FLC', 'PD', 'T1', 'T1C', 'T2']))}
class_to_int = {c: i for i, c in int_to_class.items()}


tick_size = 20
sns.set(rc={
    'figure.figsize':(10,10), 
    'font.size': 25, 
    "axes.labelsize":25, 
    "xtick.labelsize": tick_size, 
    "ytick.labelsize": tick_size,
    'font.family':'serif',
    'grid.linestyle': '',
    'axes.facecolor': 'white',
    'axes.edgecolor': '0.2',
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'axes.spines.right': True,
    'axes.spines.top': True,
})

palette = sns.color_palette("Set2", n_colors=6, desat=1)


def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def prepare_data(x_filename, target_dims, class_to_int):
    # split image into slices
    x = nib.load(x_filename).get_fdata()
    x_slices = get_axial_slices(x, target_dims)

    y = tf.one_hot(class_to_int[x_filename.parts[-2]], depth=len(class_to_int))

    return x_slices, y


def get_spaced_slices(num_slices, num_plots, offset):
    t = (num_slices-(offset*2)) // num_plots
    return [x for x in range(offset, num_slices-offset, t)]




if __name__ == "__main__":

    ########## HYPERPARAMETER SETUP ##########

    instance_size = (256, 256)
    num_classes = 6
    progbar_length = 10

    ########## DIRECTORY SETUP ##########

    if len(sys.argv) < 2:
        print("Error: missing filename argument")
        sys.exit()
    FILENAMES_FILE = sys.argv[1]
    with open(FILENAMES_FILE, 'r') as f:
        lines = [l.strip().split(',') for l in f.readlines()]
    filenames_labels = [(Path(s), l) for (s, l) in lines]

    cur_fold = sys.argv[2]

    GPUID = sys.argv[3]
    os.environ['CUDA_VISIBLE_DEVICES'] = GPUID

    MODEL_NAME = "phinet"
    WEIGHT_DIR = Path("models/weights") / MODEL_NAME
    RESULTS_DIR = Path("results")
    FIGURE_DIR = RESULTS_DIR / "figures" / "gradcams"

    # files and paths
    for d in [WEIGHT_DIR, RESULTS_DIR, FIGURE_DIR]:
        if not d.exists():
            d.mkdir(parents=True)

    MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")

    ######### MODEL #########
    with open(MODEL_PATH) as json_data:
        model = tf.keras.models.model_from_json(json.load(json_data))
    #print(model.summary(line_length=75))
    TRAINED_WEIGHTS_FILENAME = WEIGHT_DIR / "best_weights_fold_{}.h5"
    model.load_weights(str(TRAINED_WEIGHTS_FILENAME).format(cur_fold))

    ######### GRADCAM PREPARATION ######### 

    LAYER_NAME = "concatenate"
    model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(LAYER_NAME).output, model.output],
    )


    FOLD_RESULTS_FILE = RESULTS_DIR / "test_metrics_on_nifti_ALL_DATA_fold_{}.csv"
    
    # metrics
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # store corresponding scores
    x_name = []
    y_true = []
    y_pred = []

    num_elements = len(filenames_labels)

    ######### INFERENCE #########
    print()

    # write head
    with open(str(FOLD_RESULTS_FILE).format(cur_fold), 'w') as f:
        f.write("{},{},{},{}\n".format(
            "filename",
            "true_class",
            "pred_class",
            "pred_score",
        ))

    # get input file and target label
    for i, (filename, label) in tqdm(enumerate(filenames_labels), 
            total=len(filenames_labels)):

        x, y = prepare_data(filename, instance_size, class_to_int)

        with tf.GradientTape() as tape:
            target_feature_maps, logits = model(x, training=False)

            if logits.shape[-1] != y.shape[0]:
                print("\nShape mismatch.\tLogits: {}\tY: {}".format(
                        logits.shape,
                        y.shape
                    )
                )
                logits = -1.0 * tf.ones_like(y, dtype=tf.float32)

        preds = tf.nn.softmax(logits)
        pred = tf.reduce_mean(
            tf.nn.softmax(logits),
            axis=0,
        )
        
        # GradCAM
        grads = tape.gradient(logits, target_feature_maps)
        try:
            pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        except ValueError:
            continue

        CLASS_DIR = FIGURE_DIR / int_to_class[tf.argmax(y).numpy()]
        if not CLASS_DIR.exists():
            CLASS_DIR.mkdir(parents=True)

        num_figs = 25
        offset = 10
        side = int(np.sqrt(num_figs))
        fig, axs = plt.subplots(
            side, 
            side,
        )

        plt.suptitle("True: {} | Pred: {}".format(
            int_to_class[tf.argmax(y).numpy()],
            int_to_class[tf.argmax(pred).numpy()],
            ),
            y = 0.925,
        )

        spaced_slices = get_spaced_slices(x.shape[0], num_figs, offset)

        grid_indices = [(r, c) for r in range(5) for c in range(5)]

        for slice_idx, (row, col) in zip(spaced_slices, grid_indices):
            if x[slice_idx].sum() != 0:
                heatmap = tf.reduce_mean(
                    target_feature_maps[slice_idx] * pooled_grads[slice_idx],
                    axis=-1,
                )
                heatmap = np.array(tf.nn.relu(heatmap))
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                heatmap = cv2.resize(
                    heatmap, 
                    instance_size, 
                    interpolation=cv2.INTER_LINEAR,
                )

                axs[row,col].imshow(x[slice_idx, :, :, 0].T, cmap='Greys_r', vmin=0, vmax=255)
                axs[row,col].imshow(heatmap.T, cmap='plasma', vmin=0, vmax=1, alpha=0.5)
            else:
                axs[row,col].imshow(np.zeros_like(x[slice_idx, :, :, 0].T), cmap='Greys_r')
            axs[row,col].set_xticks([])
            axs[row,col].set_yticks([])

        gap = 1e-4
        plt.subplots_adjust(wspace=gap, hspace=gap)

        figure_name = CLASS_DIR / "{}_gradcam_montage.png"\
                .format(filename.name.split('.')[0])
        plt.savefig(figure_name, bbox_inches="tight")
        plt.close()



        with open(str(FOLD_RESULTS_FILE).format(cur_fold), 'a') as f:
            # write current predictions
            f.write("{},{},{}".format(
                filename,
                np.argmax(y),
                np.argmax(pred),
            ))
            f.write(",[")
            for class_pred in pred: 
                f.write("{:.2f} ".format(class_pred))
            f.write("]\n")
