#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np
import os


# In[9]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[10]:


eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[11]:


cap = cv2.VideoCapture(0)


# In[13]:


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
            
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:




