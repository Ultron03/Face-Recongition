import cv2
import face_recognition

imgElon = face_recognition.load_image_file('images/elon.jpg')           # Loading image  (imgElon is just naming ,you can keep any name you want)
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)                       # Changing it from BGR to RGB

imgtest = face_recognition.load_image_file('images/Screenshot (186).png')      # Loading other image (imgtest is just naming ,you can keep any name you want)
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)                       # Changing it from BGR to RGB

faceLoc = face_recognition.face_locations(imgElon)[0]                   # here we finding the face in our image(it gives location of top,right,bottom,left) and becoz we sending in single image we only get the first element of it([0])(it's an array)
encodeElon = face_recognition.face_encodings(imgElon)[0]                # here we encode the face found by above line ,print(faceLoc)  by doing this we can see that facelocation gives 4 values
cv2.rectangle(imgElon,(faceLoc[1],faceLoc[2]),(faceLoc[3],faceLoc[0]),(0,0,255),2)  # here were printing/showing the face we dectected with reactangle by giving it the facelocation points(4),then choose the colour of the box and it's width.

faceLoca = face_recognition.face_locations(imgtest)[0]                   # here we finding the face in our image(it gives location of top,right,bottom,left) and becoz we sending in single image we only get the first element of it([0])
encodetest = face_recognition.face_encodings(imgtest)[0]                # here we encode the face found by above line ,print(faceLoc)  by doing this we can see that facelocation gives 4 values
cv2.rectangle(imgtest,(faceLoca[1],faceLoca[2]),(faceLoca[3],faceLoca[0]),(0,0,255),2)  # here were printing/showing the face we dectected with reactangle by giving it the facelocation points(4),then choose the colour of the box and it's width.

results = face_recognition.compare_faces([encodeElon],encodetest)       # here we are comparing two images we provided and taking it as result
facDis = face_recognition.face_distance([encodeElon],encodetest)        # here we are finding the of two picture distance (less the distance the similar the picture )(it stores in form of array)(e.g.it like we are compareing length of lips and comparing it)
print(results,facDis)
cv2.putText(imgtest,f'{results}{round(facDis[0],2)}',(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2)  # here we are directly writing the result on the test img(the second img which we were comparing) ,roundingoff the faceDistance ,giving font details (origin,fontType,Scale,fontColor,width)

cv2.imshow('Elon bhaiya',imgElon)                                       # Giving the name to the img to show when see
cv2.imshow('ek aur Elon bhaiya',imgtest)                                # Giving the name to the img to show when see
cv2.waitKey(0)                                                          # Giving delay of 0 sec.


