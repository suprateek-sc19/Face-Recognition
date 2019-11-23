# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:32:17 2019

@author: Suprateek Chatterjee
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

nose_cascade = cv2.CascadeClassifier("Nose18x15.xml")
eyes_cascade = cv2.CascadeClassifier("frontalEyes35x16.xml")

while True:
    
    
    ret,frame = cap.read()
    

    if ret==False:
        continue
    
    noses = nose_cascade.detectMultiScale(frame,1.3,5)
    for nose in noses[-1:]:
        x,y,w,h = nose
        
        offset = 10
        #Extract the nose
        nose_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        nose_section = cv2.resize(nose_section,(100,100))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
    
    eyes = eyes_cascade.detectMultiScale(frame,1.3,5)
    for eye in eyes[-1:]:
        a,b,c,d = eye
        
        
        #Extract the eyes
        #eye_section = frame[b-offset:b+d+offset,a-offset:a+c+offset]
        #eye_section = cv2.resize(eye_section,(100,100))
        
        cv2.rectangle(frame,(a,b),(a+c,b+d),(0,255,255),2)
        
        
        cv2.imshow("Nose",frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()