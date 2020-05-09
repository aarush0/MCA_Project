# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:41:45 2020

@author: Aarush
"""

import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
import numpy as np    # for mathematical operations
import os

video_path = 'C:\\Users\\Aarush\\Downloads\\MELD.Raw\\MELD.Raw\\dev\\dev_splits_complete\\'
video_save_path = 'C:\\Users\\Aarush\\Downloads\\MELD.Raw\\MELD.Raw\\dev\\dev_frames\\'
video_names = os.listdir(video_path)

#print(len(os.listdir(video_save_path)))

for i, vid in enumerate(video_names):
    try:
      count = 0

      videoFile = video_path + video_names[i]

      cap = cv2.VideoCapture(videoFile)   

      #frameRate = cap.get(5) 
      frameRate = 20
      #print("fr", frameRate)
      x=1
      while(cap.isOpened()):
          frameId = cap.get(1) #current frame number
          #print(frameId)
          ret, frame = cap.read()
          if (ret != True):
              break
          if (frameId % math.floor(frameRate) == 0):
              # storing the frames in a new folder named train_1
              filename = video_save_path+vid[:vid.find('.')] + "_frame%d.jpg" % count;count+=1
              cv2.imwrite(filename, frame)
              #print(frameId, frameRate)
      cap.release()
      print(i, vid, count)
      
    except:
      print("Error in "+vid)