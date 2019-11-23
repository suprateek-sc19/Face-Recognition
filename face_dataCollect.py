# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 19:21:55 2019

@author: Suprateek Chatterjee
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './data/'
filename = input("Enter the name of the person : ")
while True:
    
    
    ret,frame = cap.read()
    

    if ret==False:
        continue
    
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    face_sorted = sorted(faces,key=lambda f:f[2]*f[3])
    
    #Pick the last face as it is largest accoding to area
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        offset = 10
        #Extract the face
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        skip += 1
        if(skip%10==0):
            face_data.append(face_section)
            print(len(face_data))
        
        
    cv2.imshow("Frame",frame)
    cv2.imshow("Face_section",face_section)
    
    if(skip%10==0):
        pass
        
    key_pressed = cv2.waitKey(1) & 0xFF
    
    if key_pressed == ord('q'):
        break
    
    
#Convert data into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)


#Save this data in file
np.save(dataset_path+filename+'.npy',face_data)
print("Data succesfully saved at"+dataset_path+filename+'.npy')
    
cap.release()
cv2.destroyAllWindows()