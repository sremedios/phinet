#!/bin/bash

if [ $# -lt 2 ];then
    echo "Usage: 
    $0 Input_Image Reorient_code Output_Image
    Input_Image     NIFTI(.nii) image
    Reorient_code   3 digit reorientation code, like RAI, ASL, RSP
    Output_Image    Output image. If not mentioned, input image is overwritten."
    exit 1
fi

tmpdir=`mktemp -d`
SUB=$1
ORI=$2
OUT=$3

SUB=`readlink -f "$SUB"`
ORI2=`3dinfo -orient $SUB`
if [ x"$OUT" == "x" ];then
    echo "*** WARNING: Input image ($SUB,$ORI2) will be overwritten."
else
    OUT=`readlink -f "$OUT"`    
fi
cd $tmpdir
fslchfiletype NIFTI "$SUB" ./t1.nii
3dresample -orient $ORI -inset ./t1.nii -prefix ./t1_reorient.nii
if [ x"$OUT" == "x" ];then
    fslchfiletype NIFTI  t1_reorient.nii "$SUB"
else    
    fslchfiletype NIFTI t1_reorient.nii  "$OUT"
fi   
rm -rf $tmpdir   
