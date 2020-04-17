



# TODO: figure out format for classification outputs, then parse into CM




import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix
import sys
import os
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

# In[45]:


dataset_counts = [672, 500, 400, 300, 200, 100]
folds = [0, 1, 2, 3, 4]
# init stats dict
stats = {
    d:{
        f: {
            y: [] for y in ["y_true", "y_pred", "y_prob"]} 
        for f in folds} 
    for d in dataset_counts
}


# # Read predictions files

# In[46]:


for dataset_count in dataset_counts:
    for cur_fold in folds:
        pred_filepath = os.path.join(
            'predictions', 
            'figure_3_dataset_{}'.format(dataset_count), 
            'model_predictions_{}_fold_{}'.format(dataset_count, cur_fold)
        )
        with open(pred_filepath, 'r') as f:
            lines = [l.strip().split(',') for l in f.readlines()]
            
        for l in lines[1:]:
            stats[dataset_count][cur_fold]["y_true"].append(int(l[0]))
            stats[dataset_count][cur_fold]["y_pred"].append(int(l[1]))
            stats[dataset_count][cur_fold]["y_prob"].append(float(l[2]))
            
        stats[dataset_count][cur_fold]["y_true"] = np.array(stats[dataset_count][cur_fold]["y_true"])
        stats[dataset_count][cur_fold]["y_pred"] = np.array(stats[dataset_count][cur_fold]["y_pred"])
        stats[dataset_count][cur_fold]["y_prob"] = np.array(stats[dataset_count][cur_fold]["y_prob"])

# # Confusion Matrix

# In[60]:


# take average of all fold stats:
dataset_count = 672
y_true = []
y_pred = []

for cur_fold in range(5):
    y_true.append(stats[dataset_count][cur_fold]["y_true"])
    y_pred.append(stats[dataset_count][cur_fold]["y_pred"])


y_true = np.array(y_true)
y_true = np.mean(y_true, axis=0)
y_true = np.round(y_true).astype(np.uint8)

y_pred = np.array(y_pred)
y_pred = np.mean(y_pred, axis=0)
y_pred = np.round(y_pred).astype(np.uint8)


# In[61]:


class_names = ['Negative', 'Positive']
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(
    cm_norm, 
    interpolation='nearest', 
    cmap=ListedColormap([
        sns.color_palette("BuGn_r", n_colors=6, desat=1)[-1],
        sns.color_palette("Set2", n_colors=6, desat=0.8)[::-1][-1],
    ]),
)
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=class_names,
    yticklabels=class_names,
    #title='Model {} Confusion Matrix\n'.format(dataset_count),
    ylabel='True Label',
    xlabel='Predicted Label',
)

labelpad = 10
ax.xaxis.labelpad = labelpad
ax.xaxis.set_tick_params(pad=labelpad)
ax.yaxis.labelpad = labelpad
ax.yaxis.set_tick_params(pad=labelpad)

#plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
plt.setp(ax.get_yticklabels(), rotation=90, ha='center', rotation_mode='anchor')

thresh = cm_norm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, "{}\n{:.2f}%".format(cm[i, j], cm_norm[i, j] * 100),
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else 'black',
                fontsize=25)
fig.tight_layout()
plt.savefig('figures/figure_3_dataset_{}/dataset_{}_Normalized_Confusion_Matrix.png'.format(dataset_count, dataset_count), bbox_inches='tight')
#plt.savefig('figures/figure_3_dataset_{}/dataset_{}_Normalized_Confusion_Matrix.eps'.format(dataset_count, dataset_count), bbox_inches='tight')


# In[70]:


dataset_counts = [672, 500, 400, 300, 200, 100]
dataset_counts.reverse() # reverse for aesthetics
manual_thresholds = np.linspace(0, 1, 1000)

fig = plt.figure(figsize=(20, 10))
plt.xlabel('Recall', labelpad=25)
plt.ylabel('Precision', labelpad=25)
#plt.title('Precision-Recall Curves\n')

for i, dataset_count in enumerate(dataset_counts):

    mean_precision = []
    mean_recall = []
    stds = []
    
    for t in manual_thresholds:
        manual_precision = []
        manual_recall = []
        
        for cur_fold in range(5): 
            
            y_true = stats[dataset_count][cur_fold]["y_true"]
            y_pred = stats[dataset_count][cur_fold]["y_pred"]
            y_prob = stats[dataset_count][cur_fold]["y_prob"]
            
#             y_true = []
#             y_pred = []
#             y_prob = []

#             pred_filepath = os.path.join(
#                 'figures', 
#                 'figure_3_dataset_{}'.format(dataset_count), 
#                 'model_predictions_{}_fold_{}'.format(dataset_count, cur_fold)
#             )
#             with open(pred_filepath, 'r') as f:
#                 lines = [l.strip().split(',') for l in f.readlines()]
#             for l in lines[1:]:
#                 y_true.append(int(l[0]))
#                 y_pred.append(int(l[1]))
#                 y_prob.append(float(l[2]))

            #precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            #average_precision = average_precision_score(y_true, y_prob)

            num_tp = 0
            num_fp = 0
            num_tn = 0
            num_fn = 0

            for yt, yp in zip(y_true, y_prob):
                if yp > t:
                    thresh_y = 1
                else:
                    thresh_y = 0
                    
                if yt == 0 and thresh_y == 0:
                    num_tn += 1
                elif yt == 0 and thresh_y == 1:
                    num_fp += 1
                elif yt == 1 and thresh_y == 1:
                    num_tp += 1
                elif yt == 1 and thresh_y == 0:
                    num_fn += 1
                    
            if num_tp == 0:
                manual_precision.append(0)
                manual_recall.append(0)
            else:    
                manual_precision.append(num_tp/(num_tp+num_fp))
                manual_recall.append(num_tp/(num_tp+num_fn))
                
            
        mean_precision.append(np.mean(manual_precision))
        mean_recall.append(np.mean(manual_recall))
    
    
    std = np.std(mean_precision, axis=0)
    step_kwargs = {'step': 'post'}
    #plt.plot(mean_recall, mean_precision, label='Model {}'.format(dataset_count))
    plt.step(mean_recall, mean_precision, color=palette[i], alpha=1, where='post', label='Model {}'.format(dataset_count), )
    plt.fill_between(mean_recall, mean_precision - std, mean_precision + std, color=palette[i], alpha=0.2, **step_kwargs)

    plt.legend()
    labelpad = 10
    plt.xlabel('Recall', labelpad=labelpad)
    plt.ylabel('Precision', labelpad=labelpad)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.01])
    

plt.savefig('figures/cv_pr_cmp.png', bbox_inches='tight')
plt.show()


# In[ ]:




