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


./phinet/
+-- data/
|   +-- train/
|   |   +-- /class\_1
|   |   |   +-- file\_1\_1.nii.gz
|   |   |   +-- file\_1\_2.nii.gz
|   |   |   +-- file\_1\_3.nii.gz
|   |   +-- /class\_2
|   |   |   +-- file\_2\_1.nii.gz
|   |   |   +-- file\_2\_2.nii.gz
|   |   |   +-- file\_2\_3.nii.gz
|   |   +-- [...]
|   |   +-- /class\_n
|   |   |   +-- file\_n\_1.nii.gz
|   |   |   +-- file\_n\_2.nii.gz
|   |   |   +-- file\_n\_3.nii.gz
|   +-- validation/
|   |   +-- /class\_1
|   |   |   +-- file\_1\_1.nii.gz
|   |   |   +-- file\_1\_2.nii.gz
|   |   |   +-- file\_1\_3.nii.gz
|   |   +-- /class\_2
|   |   |   +-- file\_2\_1.nii.gz
|   |   |   +-- file\_2\_2.nii.gz
|   |   |   +-- file\_2\_3.nii.gz
|   |   +-- [...]
|   |   +-- /class\_n
|   |   |   +-- file\_n\_1.nii.gz
|   |   |   +-- file\_n\_2.nii.gz
|   |   |   +-- file\_n\_3.nii.gz
|   +-- test/
|   |   +-- file\_1\_1.nii.gz
|   |   +-- file\_1\_2.nii.gz
|   |   +-- file\_1\_3.nii.gz
|   |   +-- file\_2\_1.nii.gz
|   |   +-- file\_2\_2.nii.gz
|   |   +-- file\_2\_3.nii.gz
|   |   +-- file\_n\_1.nii.gz
|   |   +-- file\_n\_2.nii.gz
|   |   +-- file\_n\_3.nii.gz

### Image Preprocessing
All images will be run under fsl robustfov during the loading of images, and
saved into a temporary "robustfov/" directory.  This directory will be destroyed
at the end of training, validation, or testing.

### Training
Run train.py.
### Classify
Run validate.py to get an accuracy score over data for which the labels are known.
This runs the latest model over the holdout set.

Run classify.py (TODO) to obtain mappings between filenames and classes.  This can
then be used to either validate a pipeline, automatically sort a directory, etc.

### Results from downsampled data (SPIE conference paper)
accuracy, training time, testing time

### Improved, Current Results from 3D patches
accuracy, training time, testing time

### References
If this is used, please cite this paper {TODO}
