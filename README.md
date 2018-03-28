# Phi-Net: 3D Convolutional Neural Network Implemented with Keras
## Background
Classifying modalities in magnetic resonance brain imaging with deep learning.

## Directions:
### Directory Setup
Create data directories and subdirectories as below. Training will be 
executed over the data in the train directory and validated over data in 
the validation directory.

The test directory is for images we don't know the labels for and want to
classify, for example for use in a pipeline or for images fresh from the
scanner.

```
./phinet/
+-- data/
|   +-- train/
|   |   +-- /task_1
|   |   |   +-- /class_1
|   |   |   |   +-- t_file_1_1.nii.gz
|   |   |   |   +-- t_file_1_2.nii.gz
|   |   |   |   +-- t_file_1_3.nii.gz
|   |   |   +-- /class_2
|   |   |   |   +-- t_file_2_1.nii.gz
|   |   |   |   +-- t_file_2_2.nii.gz
|   |   |   |   +-- t_file_2_3.nii.gz
|   |   |   +-- [...]
|   |   |   +-- /class_n
|   |   |   |   +-- t_file_n_1.nii.gz
|   |   |   |   +-- t_file_n_2.nii.gz
|   |   |   |   +-- t_file_n_3.nii.gz
|   |   +-- /task_2
|   |   |   +-- /class_1
|   |   |   |   +-- t_file_1_1.nii.gz
|   |   |   |   +-- t_file_1_2.nii.gz
|   |   |   |   +-- t_file_1_3.nii.gz
|   |   |   +-- /class_2
|   |   |   |   +-- t_file_2_1.nii.gz
|   |   |   |   +-- t_file_2_2.nii.gz
|   |   |   |   +-- t_file_2_3.nii.gz
|   |   |   +-- [...]
|   |   |   +-- /class_n
|   |   |   |   +-- t_file_n_1.nii.gz
|   |   |   |   +-- t_file_n_2.nii.gz
|   |   |   |   +-- t_file_n_3.nii.gz
|   |   +-- /task_n
|   |   |   +-- /class_1
|   |   |   |   +-- t_file_1_1.nii.gz
|   |   |   |   +-- t_file_1_2.nii.gz
|   |   |   |   +-- t_file_1_3.nii.gz
|   |   |   +-- /class_2
|   |   |   |   +-- t_file_2_1.nii.gz
|   |   |   |   +-- t_file_2_2.nii.gz
|   |   |   |   +-- t_file_2_3.nii.gz
|   |   |   +-- [...]
|   |   |   +-- /class_n
|   |   |   |   +-- t_file_n_1.nii.gz
|   |   |   |   +-- t_file_n_2.nii.gz
|   |   |   |   +-- t_file_n_3.nii.gz
|   +-- validation/
|   |   +-- /task_1
|   |   |   +-- /class_1
|   |   |   |   +-- v_file_1_1.nii.gz
|   |   |   |   +-- v_file_1_2.nii.gz
|   |   |   |   +-- v_file_1_3.nii.gz
|   |   |   +-- /class_2
|   |   |   |   +-- v_file_2_1.nii.gz
|   |   |   |   +-- v_file_2_2.nii.gz
|   |   |   |   +-- v_file_2_3.nii.gz
|   |   |   +-- [...]
|   |   |   +-- /class_n
|   |   |   |   +-- v_file_n_1.nii.gz
|   |   |   |   +-- v_file_n_2.nii.gz
|   |   |   |   +-- v_file_n_3.nii.gz
|   |   +-- /task_n
|   |   |   +-- /class_1
|   |   |   |   +-- v_file_1_1.nii.gz
|   |   |   |   +-- v_file_1_2.nii.gz
|   |   |   |   +-- v_file_1_3.nii.gz
|   |   |   +-- /class_2
|   |   |   |   +-- v_file_2_1.nii.gz
|   |   |   |   +-- v_file_2_2.nii.gz
|   |   |   |   +-- v_file_2_3.nii.gz
|   |   |   +-- [...]
|   |   |   +-- /class_n
|   |   |   |   +-- v_file_n_1.nii.gz
|   |   |   |   +-- v_file_n_2.nii.gz
|   |   |   |   +-- v_file_n_3.nii.gz
|   +-- test/
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- file_3.nii.gz
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- file_3.nii.gz
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- file_3.nii.gz
```
### Training
Usage:

Run `train.py` on a task with some desired arguments.
The three tasks are:
- modality: Classifying a T1, T2, or FLAIR image without contrast
- t1-contrast: Classifying as T1 pre-contrast or T1 post-contrast
- fl-contrast: Classifying as FLAIR pre-contrast or FLAIR post-contrast

`--task`: Type of task, one of:
            - modality
            - t1-contrast
            - fl-contrast

`--datadir`: Path to where the unprocessed data is

`--o`: Path to location where the weights will be saved


Example usage:
`python train.py --task modality --traindir data/train/ --o weights/modality/` 

### Classify
Usage:

Run `predict.py` with some desired arguments:

`--task`: Type of task, one of:
            - modality
            - t1-contrast
            - fl-contrast

`--infile`: path to the file to to classify

`--model`: path to the neural network model to use

`--o`: path and filename where results are written

`--preprocesseddir`: output directory where final preprocessed image will be placed

Example usage:
`python predict.py --infile data/test/my_brain.nii.gz --model weights/modality/my_weights.hdf5 --o myresults.txt --preprocesseddir data/test/preprocess`

### Validate
Usage: validate the model on some test data and record metrics.

Run `validate.py` with some desired arguments:

`--task`: Type of task, one of:
            - modality
            - t1-contrast
            - fl-contrast

`--datadir`: Path to where the unprocessed data is

`--model`: path to the neural network model to use

`--o`: path to directory where results are written

Example usage:
`python validate.py --task modality --datadir data/validation/ --model weights/modality/my_weights.hdf5 --o validation_results/`

### Image Preprocessing
Here are all the preprocessing steps which are automatically executed in `train.py`, `validate.py`, and `test.py`.

All preprocessing code is located in `utils/utils.py`.

First, all images are converted to 256x256x256 at 1mm^3 with intensities in [0,255]
using FreeSurfer's `mri_convert`.

Then all images are rotated into RAI orientation using AFNI `3dresample`.  While unnecessary,
this allows for visual inspection of images learned by the model.

Then each of these images will be run under fsl's `robustfov` to remove the necks.

Finally all images are run under `3dWarp`, which aligns the images as well as downsamples them
to 2mm^3 if necessary for RAM constraints.


### Results from downsampled data (SPIE conference paper)
{TODO}
accuracy, training time, testing time

### Improved, Current Results from 3D patches
{TODO}
accuracy, training time, testing time

### References
The associated paper is available on ResearchGate: https://www.researchgate.net/publication/323440662_Classifying_magnetic_resonance_image_modalities_with_convolutional_neural_networks

If this is used, please cite our work:
Samuel Remedios, Dzung L. Pham, John A. Butman, Snehashis Roy, "Classifying magnetic resonance image modalities with convolutional neural networks," Proc. SPIE 10575, Medical Imaging 2018: Computer-Aided Diagnosis, 105752I (27 February 2018); doi: 10.1117/12.2293943
