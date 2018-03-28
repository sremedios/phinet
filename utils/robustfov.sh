#!/bin/bash
if [ $# -lt 1 ];then
echo "Usage:
./robustfov.sh INPUT_IMAGE OUTPUT_IMAGE LENGTH
If OUTPUT_IMAGE is not mentioned, the output image is written in the same directory as the INPUT_IMAGE
with an extension of _robustfov

Input and output image must be nifti (.nii). nii.gz not accepted.

LENGTH is the size of the brain in z-dimensiion. Default is 170 (in mm). "
  exit 1
fi

SUB=$1
OUT=$2
LEN=$3
SUB=`readlink -f $SUB`
if [ x"$OUT" != "x" ];then
  OUT=`readlink -f $OUT`
else
  OUT=`remove_ext $SUB`
  OUT="${OUT}"_robustfov.nii
fi
if [ x"$LEN" == "x" ];then
	LEN=170
fi

FSLOUTPUTTYPE=NIFTI
workingdir=`mktemp -d`
cd $workingdir
fslchfiletype NIFTI $SUB ./t1.nii
echo robustfov -b $LEN -i t1.nii -r temp.nii -m temp.mat
robustfov -b $LEN -i t1.nii -r temp.nii -m temp.mat &>/dev/null
flirt -in temp.nii -ref t1.nii -applyxfm -init temp.mat -out t1_robustfov.nii

if [ ! -f "$OUT" ];then
  mv -vf t1_robustfov.nii $OUT
else
  echo "Output file ($OUT) already exists. I will not overwrite. Making a copy instead."
  ID=`remove_ext $OUT`
#  ID=${OUT%.*}
  ID="${ID}"_robustfov.nii
  mv -vf t1_robustfov.nii $ID
fi
cd
rm -rf $workingdir
