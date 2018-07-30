#!/usr/bin/env /ISFILE/APPS/ECdata/pipeline/final/bin/python
# -*- coding: utf-8 -*-

import os
from subprocess import Popen
from shutil import move
from multiprocessing.pool import ThreadPool

rootdir = '/home/USERS/olsonjd/register_test/'


def nii2rai(i, infile):
    cmd = '3dresample -orient rai -input ' + infile + ' -prefix ' + \
        ''.join(infile.split('.')[:-2]) + '_rai.nii.gz'
    print(i, cmd)
    Popen(cmd.split()).wait()
    os.remove(infile)
    move(''.join(infile.split('.')[:-2]) + '_rai.nii.gz', infile)

filelist = []
for root, dirs, files in os.walk(rootdir):
    for f in sorted(files):
        if f.endswith('nii.gz'):
            filelist.append(os.path.join(root, f))
            # anz2nii(os.path.join(root, f))

tp = ThreadPool(30)
for i, f in enumerate(filelist):
    tp.apply_async(nii2rai, (i, f,))

tp.close()
tp.join()
