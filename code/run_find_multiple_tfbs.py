# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 17:19:52 2022

@author: z's'q
"""

import os
path = '../../multiple_mmgraph2/data/encode_101M'
files = os.listdir(path)
data = [t.split('.')[0][:-2] for t in files if 'B' in t]

for i in range(0,len(data)):
    name = data[i]
    CMD = 'python find_multiple_tfbs.py --dataset ' + name
    os.system(CMD)
