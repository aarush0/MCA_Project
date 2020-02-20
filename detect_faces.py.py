#!/usr/bin/env python

import numpy as np
import cv2
import ntpath
import glob

# local modules
from video import create_capture
from common import clock, draw_str


help_message = '''
USAGE: facedetect.py [--cascade <cascade_fn>] [--nested-cascade <nested_fn>] [<video_source>]
'''

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    samples = 30

    path = 'C:/Users/Aarush/Desktop/MUStARD-master/data/videos'
    files = glob.glob(path+ "/*.mp4")
    
    for fn in files:   

        try:

            if fn.startswith('face'):
                continue

            s = fn.rfind('\\')
            s1= fn[:s] + '/' + fn[s+1:]
            fn = s1
            print(fn)

            args, video_src = getopt.getopt(fn, '', ['cascade=', 'nested-cascade='])
            args = dict(args)
            cascade_fn = args.get('--cascade', "./haarcascade_frontalface_alt2.xml")
            nested_fn  = args.get('--nested-cascade', "./haarcascade_eye.xml")
        
            cascade = cv2.CascadeClassifier(cascade_fn)
            nested = cv2.CascadeClassifier(nested_fn)
        
            cam = create_capture(video_src)
            fps = cam.get(5)
        
            #         w=int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))
            # h =int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))
            # #  video recorder
            # f ourcc = cv2.cv.CV_FOURCC(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
            # v ideo_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))
        
            # fourcc = cv2.VideoWriter_fourcc(*'MOV')
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        
            name = ntpath.basename(fn)  
            out = cv2.VideoWriter('face_' + name , fourcc, fps, (240,320), True)
        
            while True:
                ret, img = cam.read()
                if not ret : break
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)       
                t = clock()
                
                rects = detect(gray, cascade)
                
                vis = img.copy()
                # draw_rects(vis, rects, (0, 255, 0))       
                if not nested.empty():
                    
                    for x1, y1, x2, y2 in rects:
                        roi = gray[y1:y2, x1:x2]
                        vis_roi = vis[y1:y2, x1:x2]
                        # subrects = detect(roi.copy(), nested)
                        # draw_rects(vis_roi, subrects, (255, 0, 0))
                        res = cv2.resize(vis_roi,(240, 320), interpolation = cv2.INTER_CUBIC)       
                        cv2.imshow('facedetect', res)
                        out.write(res)


            out.release()

            cv2.destroyAllWindows()

        except:
            continue

    
        # if 0xFF & cv2.waitKey(5) == 27: break   



            