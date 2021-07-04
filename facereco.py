
import cv2
import numpy as np
import face_recognition
import os

path = "Images Reco"
images = []
className = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    className.append(os.path.splitext(cls)[0])
# print(images)
print(className)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

findEncode = findEncodings(images)
print("Encoding Complete")


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0),None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for faceEnco, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(findEncode, faceEnco)
        faceDis = face_recognition.face_distance(findEncode, faceEnco)
        # print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name, (x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

# img_me = face_recognition.load_image_file("Images Reco/Priya_1.jpg")
# img_me = cv2.resize(img_me, (350,330))
# img_me = cv2.cvtColor(img_me, cv2.COLOR_BGR2RGB)
#
# cv2.imshow("Priya", img_me)
#
# img_didi = face_recognition.load_image_file("Images Reco/Didi_1.jpg")
# img_didi = cv2.resize(img_didi, (350,450))
# img_me = cv2.cvtColor(img_me, cv2.COLOR_BGR2RGB)
# cv2.imshow("didi", img_didi)
#
# cv2.waitKey(0)