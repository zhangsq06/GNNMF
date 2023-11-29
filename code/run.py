# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:35:06 2021

@author: ShuangquanZhang
"""
import os
path = '../../multiple_mmgraph2/data/encode_101H'
files = os.listdir(path)
names = [t.split('.')[0] for t in files if '_B' not in t and '_AC' not in t]
# names = [t for t in names if 'top' in t ]
#############################################
for i in range(0,len(names)):
    print(names[i])
    cmd = 'python train.py --dataset '+names[i] + ' --k 5'
    os.system(cmd)