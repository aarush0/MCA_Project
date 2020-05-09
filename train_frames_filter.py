# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:10:36 2020

@author: Aarush
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict


frames_path = 'C:\\Users\\Aarush\\Downloads\\MELD.Raw\\MELD.Raw\\dev\\dev_frames\\'
frames = os.listdir(frames_path)

frames_path_new = 'C:\\Users\\Aarush\\Downloads\\MELD.Raw\\MELD.Raw\\dev\\dev_frames_final\\'

test_csv = pd.read_csv(r'C:\\Users\\Aarush\\Downloads\\MELD.Raw\\MELD.Raw\\dev_sent_emo.csv')
test_csv = np.array(test_csv)

dia_frames_train = defaultdict(list)

for i in range(len(frames)):
    
    nm = frames[i]
    f_ = nm.find('_')
    l_ = nm.rfind('_')

    utt = nm[f_+4 : l_]
    dia = nm[3: f_]
    
    for t in test_csv:
      if str(t[5]) == dia and str(t[6]) == utt:
        emo = t[3]
        if emo == 'neutral' or emo == 'joy' or emo == 'anger' or emo == 'sadness':
            key = dia+"_"+utt
            dia_frames_train[key].append(i)
        break
    

fn_dict = 'C:\\Users\\Aarush\\Downloads\\MELD.Raw\\MELD.Raw\\dev\\dia_frames_dev.pkl'
outfile = open(fn_dict, 'wb')
np.save(outfile, dia_frames_train)

dia_frames_train = np.load('C:\\Users\\Aarush\\Downloads\\MELD.Raw\\MELD.Raw\\dev\\dia_frames_dev.pkl', allow_pickle=True).item()
print(dia_frames_train)

import shutil
for number, k in enumerate(dia_frames_train.keys()):
    print(number, end= " ")
    frames_list = dia_frames_train[k]
    
    skip = len(frames_list)//5
    
    u = k.find('_')
    dia = k[:u]
    utt = k[u+1:]
    
    if skip == 0:
        skip = 1
    
    for i in range(0, len(frames_list), skip):
        
        old_path = frames_path + 'dia' + dia + '_utt' + utt + '_frame' + str(i) + '.jpg'
        new_path = frames_path_new + 'dia' + dia + '_utt' + utt + '_frame' + str(i) + '.jpg' 
        
        shutil.copy(old_path, new_path)
        
    
    
    
        
