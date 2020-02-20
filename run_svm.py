# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:24:25 2020

@author: Aarush
"""
from sklearn import svm
import numpy as np

clf1 = svm.SVC(kernel = 'rbf', C = 10, gamma = 0.00001)
clf3 = svm.SVC(kernel = 'sigmoid', C = 10, gamma = 0.0000001)


for i in range(5):
    
    train_set = []
    test_set = []
    labels_train = []
    labels_test = []
    
    f = 'C:/Users/Aarush/Desktop\MUStARD-master/data/videos/face_videos/output/' + str(i) + 'train_set.pkl' 
    with open(f, 'rb') as file:
        train_set = np.load(file)
            
    f  = 'C:/Users/Aarush/Desktop\MUStARD-master/data/videos/face_videos/output/'+ str(i)+ 'test_set.pkl'
    with open(f, 'rb') as file:
        test_set = np.load(file)
        
    f = 'C:/Users/Aarush/Desktop\MUStARD-master/data/videos/face_videos/output/' + str(i)+'labels_train.pkl'
    with open(f, 'rb') as file:
        labels_train  = np.load(file)
    
    f = 'C:/Users/Aarush/Desktop\MUStARD-master/data/videos/face_videos/output/' + str(i) + 'labels_test.pkl'
    with open(f, 'rb') as file:
        labels_test = np.load(file)
        
    print("Training "+str(i))
    
    clf1.fit(train_set, labels_train)
    predicted = clf1.predict(test_set)
    err1 = (labels_test == predicted).mean()
    
    print("1")
    
    #clf2.fit(train_set, labels_train)
    #predicted = clf2.predict(test_set)
    #err2 = (labels_test == predicted).mean()
    
    #print("2")
    
    clf3.fit(train_set, labels_train)
    predicted = clf3.predict(test_set)
    err3 = (labels_test == predicted).mean()
    
    print ('accuracy svm: %.2f %%' % (err1*100))
    #print ('accuracy svm: %.2f %%' % (err2*100))
    print ('accuracy svm: %.2f %%' % (err3*100))
    


    
    