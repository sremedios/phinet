import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
import sys
import os
import shutil
from sklearn.utils import shuffle

########## PLOT PARAMETERS ########## 

tick_size = 20
matplotlib.rcParams.update({
    'figure.figsize':(20,10), 
    'font.size': 25, 
    "axes.labelsize":25, 
    "xtick.labelsize": tick_size, 
    "ytick.labelsize": tick_size,
    'font.family':'serif'
})

sns.set(rc={
    'figure.figsize':(20,10), 
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

########## DIRECTORY SETUP ########## 

MODEL_NAME = "phinet"
RESULTS_DIR = Path("./results") / MODEL_NAME
results_path = sorted(list(RESULTS_DIR.iterdir()))


def plot_ci(train_metrics, val_metrics, palette, fname, metric_name=None):
    for i, (metrics, dataset_type) in enumerate(zip((train_metrics, val_metrics), ("Train", "Validation"))):
        upper_bound = min([len(m) for m in metrics])
        trimmed_metrics = [x[:upper_bound] for x in metrics]
        
        mean_metrics = np.mean(trimmed_metrics, axis=0)
        std_metrics = np.std(trimmed_metrics, axis=0)
        
        plt.plot(mean_metrics, color=palette[i], label="{} {}".format(dataset_type, metric_name))
        plt.fill_between(range(len(mean_metrics)), 
                         mean_metrics - std_metrics, 
                         mean_metrics + std_metrics, 
                         alpha=0.2,
                         color=palette[i])

    plt.xlabel = "Epoch"
    plt.ylabel = metric_name
    plt.title("Five-fold Cross-validated Training and Validation {} Curves".format(metric_name))
    plt.legend()
    plt.save(fname)


plot_ci(train_accs, val_accs, palette, "Accuracy")
plot_ci(train_losses, val_losses, palette, "Loss")
