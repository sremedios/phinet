#!/usr/bin/env /ISFILE/APPS/ECdata/pipeline/final/bin/python
# -*- coding: utf-8 -*-

import os
from subprocess import Popen
from multiprocessing.pool import ThreadPool

rootdir = '/home/USERS/olsonjd/register_test/'


def anz2nii(infile):
    cmd = 'mri_convert -it analyze -ot nii -i ' + infile + ' ' + \
        os.path.splitext(infile)[0] + '.nii.gz'
    print(cmd)
    Popen(cmd.split()).wait()

filelist = []
for root, dirs, files in os.walk(rootdir):
    for f in sorted(files):
        if f.endswith('img'):
            filelist.append(os.path.join(root, f))
            # anz2nii(os.path.join(root, f))

tp = ThreadPool(30)
for f in filelist:
    tp.apply_async(anz2nii, (f,))

tp.close()
tp.join()
