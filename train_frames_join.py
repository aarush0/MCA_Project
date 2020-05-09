# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:08:22 2020

@author: Aarush
"""

import pandas as pd
import os
import cv2
import numpy as np
from collections import defaultdict

train_image = []
Y = []

video_save_path = 'C:\\Users\\Aarush\\Downloads\\MELD.Raw\\MELD.Raw\\dev\\dev_frames_final\\'

test_csv = pd.read_csv(r'C:\\Users\\Aarush\\Downloads\\MELD.Raw\\MELD.Raw\\dev_sent_emo.csv')
test_csv = np.array(test_csv)

dia_frames_train = defaultdict(list)

frames = os.listdir(video_save_path)

for i in range(len(frames)):
    
    nm = frames[i]
    f_ = nm.find('_')
    l_ = nm.rfind('_')

    utt = nm[f_+4 : l_]
    dia = nm[3: f_]
    
    for t in test_csv:
      if str(t[5]) == dia and str(t[6]) == utt:
        emo = t[3]
        Y.append(t[3])
        print(i, emo, end = " ")
        break

        
    key = dia+"_"+utt
    dia_frames_train[key].append(i)
        
    img = cv2.imread(video_save_path+frames[i])
    img = cv2.resize(img, (64,64))
    img = np.array(img)
    
    img = img/255
    train_image.append(img)
    
    
    

        
fn_dict = 'C:\\Users\\Aarush\\Downloads\\MELD.Raw\\MELD.Raw\\dev\\dia_frames_dev_dia_to_row.pkl'
outfile = open(fn_dict, 'wb')
np.save(outfile, dia_frames_train)

fn_x = 'C:\\Users\\Aarush\\Downloads\\MELD.Raw\\MELD.Raw\\dev\\frames_dev_final.pkl'
outfile = open(fn_x, 'wb')
np.save(outfile, train_image)

    
