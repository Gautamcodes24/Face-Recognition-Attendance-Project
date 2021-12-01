import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path="Basic_Images"
images=[]
classname=[]
myList=os.listdir(path)
print(myList)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classname.append(os.path.splitext(cl)[0]) #removing .jpg we need name only
print(classname)

def findEncoding(images): #Finding Encoding for all images
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
#mark attendance in the .cvs file
def markAttendance(name):
    with open('Marked_Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            #tdy_date=datetime.date(datetime.now())
            dString=now.strftime('%d/%m/%Y')
            Login='Login'
            f.writelines(f'\n{name},{dtString},{dString},{Login}')
            print("Attendance Marked Successfully",name)
            print(Login)



encodeListKnow=findEncoding(images)
print("Encode Successfull")


#capturing video
cap=cv2.VideoCapture(0)

while True: #to get frame one by one
    success,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)  #reducing images size
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgs)
    encodesCurFrame=face_recognition.face_encodings(imgs,facesCurFrame)  #jaha pr error aa raha tha

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame): #taking one by one and giving for comp
        matches=face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnow,encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)   # finding minimum value

        #making circal and printing name on matched photo
        if matches[matchIndex]:
            name=classname[matchIndex].upper()
            #print(name)
            #drawing rectangle over matched images
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4  #for correct location circle
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2) #0,255,0 is color code for rectangle
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            markAttendance(name) #calling attendance function
    cv2.imshow('webcame',img)
    cv2.waitKey(1)




