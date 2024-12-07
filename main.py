import os
import cv2
import pickle
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage
import numpy as np
from datetime import datetime


cred = credentials.Certificate("seviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://face-attendance-real-tim-1a5b8-default-rtdb.firebaseio.com/",
    'storageBucket':"face-attendance-real-tim-1a5b8.appspot.com"
})
bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480);

imageBackground = cv2.imread('Resources/Background3.png')
#importing mode images into a list
folderModePath = 'Resources/modes'
modePathList = os.listdir(folderModePath)
imgModeList= []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

#print(len(modePathList))
# load the encode file
print("loading encoding file")
file = open("encodeFile.p",'rb')
encodeListKnownwithids= pickle.load(file)
file.close()
encodeListKnown , StudentIds = encodeListKnownwithids
#print(StudentIds)

print ("encode file loaded")

modeType = 0
counter = 0
id = -1
imgStudent = []

while True:
    success , img = cap.read()
    imgS = cv2.resize(img , (0,0), None , 0.25 , 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


    faceCurFrame= face_recognition.face_locations(imgS)
    encodeCurFrame= face_recognition.face_encodings(imgS, faceCurFrame)


    imageBackground[187:187+480,50:50+640]=img
    imageBackground[75:75 + 625, 800:800 + 413] = imgModeList[modeType]
    if faceCurFrame:

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print("matches",matches)
            # print("faceDis",faceDis)

            matchIndex = np.argmin(faceDis)
            # print('Match Index',matchIndex)

            if matches[matchIndex]:
                # print("known face detected")
                # print(StudentIds[matchIndex])
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 50 + x1, 187 + y1, x2 - x1, y2 - y1
                imageBackground = cvzone.cornerRect(imageBackground, bbox, rt=0)
                id = StudentIds[matchIndex]

                if counter == 0:
                    cvzone.putTextRect(imageBackground,"Loading",(275,400))
                    cv2.imshow("face Attendance", imageBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1

        if counter != 0:

            if counter == 1:
                # get the data
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)
                # get the image from storage
                blob = bucket.get_blob(f'Images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                # update data of attendance
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                   "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)
                if secondsElapsed < 30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType = 3
                    counter = 0
                    imageBackground[75:75 + 625, 800:800 + 413] = imgModeList[modeType]

            if modeType!= 3:

                if 10 < counter < 20:
                    modeType = 2

                imageBackground[75:75 + 625, 800:800 + 413] = imgModeList[modeType]

                if counter <= 10:
                    studentInfo = db.reference(f'Students/{id}').get()
                cv2.putText(imageBackground, str(studentInfo['total_attendance']), (855, 135),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.putText(imageBackground, str(studentInfo['major']), (1025, 580),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (225, 225, 225), 1)
                cv2.putText(imageBackground, str(id), (1025, 525),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (225, 225, 225), 1)
                cv2.putText(imageBackground, str(studentInfo['standing']), (910, 650),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                cv2.putText(imageBackground, str(studentInfo['year']), (1025, 650),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                cv2.putText(imageBackground, str(studentInfo['starting_year']), (1125, 650),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                offset = (413 - w) // 2
                cv2.putText(imageBackground, str(studentInfo['name']), (830 + offset, 450),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (50, 50, 50), 1)

                imageBackground[200:200 + 213, 900:900 + 213] = imgStudent

            if 10 < counter < 20:
                modeType = 2

            counter += 1

            if counter >= 20:
                counter = 0
                modeType = 0
                studentInfo = []
                imgStudent = []
                imageBackground[75:75 + 625, 800:800 + 413] = imgModeList[modeType]
    else:
        modeType=0
        counter=0

    #cv2.imshow('img',img)
    cv2.imshow("face Attendance",imageBackground)
    k=cv2.waitKey(1)
    if k==27:
        break

#cap.release()
#cv2.destroyAllWindows()
