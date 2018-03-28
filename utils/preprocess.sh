#!/bin/bash


if [ $# -lt "2" ];then
	echo "./preprocess.sh IMAGE OUTPUTDIR"
	exit 1
fi
FSLOUTPUTTYPE=NIFTI
IMG=$1
OUTDIR=$2
TMPDIR=`mktemp -d`
# convert image to 256^3 1mm^3 coronal images with intensity range [0,255] 
mri_convert -c  $IMG $TMPDIR/image.nii 
# reorient to RAI. not necessary.
utils/reorient.sh $TMPDIR/image.nii RAI
# robustfov to make sure neck isn't included
utils/robustfov.sh $TMPDIR/image.nii $TMPDIR/image_robust.nii 160
# 3dWarp to make images AC-PC aligned. not necessary, only used to make all images
# aligned. Ideally they should be rigid registered to some template fo uniformity,
# but rigid registration is slow. this is faster way.
# -newgrid 2 will resample the image to 2mm^3 resolution. 
echo "3dWarp -deoblique -NN -newgrid 2  -prefix $TMPDIR/image_warp.nii.gz  $TMPDIR/image_robust.nii"
3dWarp -deoblique -NN -newgrid 2  -prefix $TMPDIR/image_warp.nii.gz  $TMPDIR/image_robust.nii
X=`basename $IMG`
X=`remove_ext $X`
mv -vf $TMPDIR/image_warp.nii.gz $OUTDIR/${X}_processed.nii.gz
rm -rf $TMPDIR
