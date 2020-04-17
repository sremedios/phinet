#!/bin/bash
if [ $# -lt "4" ]; then
echo "==========================================================================
Usage:
$0 fixed.nii.gz moving.nii.gz mysetting registeredvolume.nii.gz

fixed.nii.gz       Fixed image
moving.nii.gz      Moving image
mysetting          Either for production (slowest), fast, or fastfortesting(fastest)


To transform another image (such as label) using this transform :

1) First, run the following command to make sure the label image
   headers are same as that of moving image
   cp /home/user/labelimage.nii ./otherimage.nii
   fslcpgeom moving.nii otherimage.nii
(Copy because fslcpgeom overwrites files)
2) Then use antsApplyTransforms  to transform the label image,
antsApplyTransforms -d 3 -i otherimage.nii -r fixed.nii -o otherimage_reg.nii -n BSpline/NearestNeighbor -f 0 -v 1 
  -t registeredVolume1Warp.nii.gz -t registeredVolume0GenericAffine.mat
   
========================================================================="
exit 1
fi


red=`tput setaf 5`
green=`tput setaf 2`
reset=`tput sgr0`

dim=3 # image dimensionality
a="`hostname`"
AP=/home/USERS/roys5/Programs/ANTs/build/bin/
if [ `echo $a | grep -c "cz" ` -gt 0 ];then
  #echo "${red}On CCCZ, using older version of ANTS with 8 CPU $reset"
  #AP="/home/USERS/chouy/Applications/ANTS/antsbin/bin"
  ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
else
  #echo "${red}On CCIRON/STEEL, using older version of ANTS with 12 CPU $reset"
  #AP="/home/USERS/chouy/Applications/ANTS/antsbin/bin"
  ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=12
fi

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS
f=$1 ; m=$2    # fixed and moving image file names
mysetting=$3
outvol=$4


FSLOUTPUTTYPE=NIFTI
prefix=`remove_ext ${outvol}`
if [ -f "${prefix}.nii" ] || [ -f "${prefix}.nii.gz" ];then
    echo "$prefix exists. I will not overwrite."
    exit 1
fi


reg=${AP}/antsRegistration           # path to antsRegistration

if [[ $mysetting == "fastfortesting" ]] ; then
  its=100x50x25
  percentage=0.1
  syn="100x1x0,0.0001,4"
elif   [[ $mysetting == "forproduction" ]] ; then
  its=1000x1000x1000
#  its=10000x111110x11110
  percentage=0.3
  syn="100x100x50,0.00001,5"
elif [[ $mysetting == "fast" ]] ; then
   its=100x100x100
  percentage=0.3
  syn="20x20x10,0.00001,5"
else
    echo "${red}ERROR: setting must be either forproduction (slowest), fast, or fastfortesting(fastest). $reset"
    exit 1
fi
START=$(date +%s)
#echo affine $m $f outname is $nm, am using setting $mysetting
#nm=${D}${nm1}_fixed_${nm2}_moving_setting_is_${mysetting}   # construct output prefix

# First do affine registration, then do this, sometimes the difference can be astounding
#tmpdir=`mktemp -d`
#echo antsaffine.sh $f $m $tmpdir/tmp_moving.nii no ${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS}
#antsaffine.sh $f $m $tmpdir/tmp_moving.nii no ${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS}
#m=$tmpdir/tmp_moving.nii
echo "$reg -d $dim -r [ $f, $m ,1]  -w [ 0.01, 0.99] -u --float -m mattes[  $f, $m , 1 , 32, regular, $percentage ] -t translation[ 0.1 ] -c [$its,1.e-8,20]  -s 4x2x1vox  -f 6x4x2 -l 1 -m mattes[  $f, $m , 1 , 32, regular, $percentage ] -t rigid[ 0.1 ] -c [$its,1.e-8,20]  -s 4x2x1vox  -f 3x2x1 -l 1 -m mattes[  $f, $m , 1 , 32, regular, $percentage ] -t affine[ 0.1 ] -c [$its,1.e-8,20]  -s 4x2x1vox  -f 3x2x1 -l 1 -m mattes[  $f, $m , 0.5 , 32 ] -m cc[  $f, $m , 0.5 , 4 ] -t SyN[ .20, 3, 0 ] -c [ $syn ]  -s 1x0.5x0vox  -f 4x2x1 -l 1 -u 1 -z 1 -o [ ${prefix} ] -v 0"
$reg -d $dim -r [ $f, $m ,1]  -w [ 0.01, 0.99] -u --float \
                        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
                         -t translation[ 0.1 ] \
                         -c [$its,1.e-8,20]  \
                        -s 4x2x1vox  \
                        -f 6x4x2 -l 1 \
                        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
                         -t rigid[ 0.1 ] \
                         -c [$its,1.e-8,20]  \
                        -s 4x2x1vox  \
                        -f 3x2x1 -l 1 \
                        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
                         -t affine[ 0.1 ] \
                         -c [$its,1.e-8,20]  \
                        -s 4x2x1vox  \
                        -f 3x2x1 -l 1 \
                        -m mattes[  $f, $m , 0.5 , 32 ] \
                        -m cc[  $f, $m , 0.5 , 4 ] \
                         -t SyN[ .20, 3, 0 ] \
                         -c [ $syn ]  \
                        -s 1x0.5x0vox  \
                        -f 4x2x1 -l 1 -u 1 -z 1 \
                        -o [ ${prefix} ] -v 0



antsApplyTransforms -d $dim -i $m -r $f -o ${prefix}.nii.gz -n BSpline -f 0 -v 1 -t  "$prefix"1Warp.nii.gz -t "$prefix"0GenericAffine.mat

fslmaths $outvol -thr 0 $outvol
#rm -rf $tmpdir


END=$(date +%s)
DIFF=$(( $END - $START ))
((sec=DIFF%60, DIFF/=60, min=DIFF%60, hrs=DIFF/60))
echo "${green}ANTS deformable registration took $hrs HRS $min MIN $sec SEC ${reset}"
