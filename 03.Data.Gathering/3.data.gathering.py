import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

face_detector = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

#for each persion, enter one numeric face id

face_id = input('\n enter user id end press <return> ==> ')
print("\n [INFO] Initializing face capture. Look at the camera and wait..")

count = 0
while True:
    ret, img = cap.read()
    #img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count +=1
        #Save the captured image into the dataset foldera
        cv2.imwrite("../face.sample/User."+str(face_id)+'.'+str(count)+".jpg",gray[y:y+h,x:x+w])
        cv2.imshow('image',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    elif count >= 30: #Take 30 face sample and stop video
        break

cap.release()
cv2.destroyAllWindows()
