import os
from datetime import datetime
import cv2
import face_recognition
import numpy as np

path = 'students list'                                                 # storing the images location in path
image = []
classNames = []
myList = os.listdir(path)                                              # taking the list of images form the path
# print(myList)

for cl in myList:                                                      # took an string cl and put it into loop for all value in myList
    curImg = cv2.imread(f'{path}/{cl}')                                # taking th path and name(cl) of  current image and storing it into curImg
    image.append(curImg)                                               # appending the img form curImg
    classNames.append(os.path.splitext(cl)[0])                         # Appending the name of img ,splitting the name with .jpeg,jpg ot any other thing and taking it's starting string value

# print(classNames)
# Created an Function to encode all the img in the list
def findEncodings(image):
    encodeList = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                     # converting every img form the image from RGB to BGR
        encode = face_recognition.face_encodings(img)[0]               # Encoding every img
        encodeList.append(encode)                                      # now adding the encoded image to encodeList
    return encodeList

def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            # date = datetime.date
            date = now.date()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString},{date}')



encodeListKnown = findEncodings(image)
print('Encoding Complete')                                                              # Because Encoding takes time we are printing this to let us know

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()                                                             # This will give our img
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)                                         # To increase the speed of the Process reduce the size of img
    imgS = cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR)

    faceCurFrame = face_recognition.face_locations(imgS)                                   # there can be multiple face in the webcam so to find the location all faces and send it to encoding
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)                   # encoding the faces Visible in webcam

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):                            # becoz we want encodeCurFrame & faceCurFrame in same loop we used zip
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)              # Comparing one of the Encoded face from the webcam to the Known Encoded List
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)               # Finding the distance between the faces
        #print(faceDis)
        matchIndex = np.argmin(faceDis)                                                    # after comparing the distance with all the images here we are storing the lowest distance we found

        if matches[matchIndex]:                                                          # If img matches , take there distance
            name = classNames[matchIndex].upper()                                         # here we storing the name of img who's distance we found minimum
           # print(name)
            y1,x2,y2,x1 = faceLoc                                                         # Storing the 4 points of Face Loction in 4 points
            y1, x2,y2, x1 = y1*4, x2*4,y2*4,x1*4                                          # y1*4, x2*4,y2*4, x1*4                             # in line 33 we reduce the size of original picture to 1/4th ,so we multipling the point(faceLoaction) by 4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)




