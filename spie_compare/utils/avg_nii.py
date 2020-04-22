import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

src_dir = Path(sys.argv[1])
dst_dir = Path(sys.argv[2])
if not dst_dir.exists():
    dst_dir.mkdir(parents=True)

fnames = sorted([x for x in src_dir.iterdir() if '.nii' in x.suffixes])
obj = nib.load(fnames[0])

avg_atlas = np.zeros(obj.shape, dtype=np.float32)

for i, fname in tqdm(enumerate(fnames), total=len(fnames)):
    avg_atlas = running_average(
            avg_atlas,
            nib.load(fname).get_fdata(dtype=np.float32),
            i+1,
        )
avg_obj = nib.Nifti1Image(avg_atlas, obj.affine)
nib.save(avg_obj, dst_dir/"atlas.nii.gz")
