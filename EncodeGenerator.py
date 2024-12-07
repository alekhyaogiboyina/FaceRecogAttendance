import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("seviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://face-attendance-real-tim-1a5b8-default-rtdb.firebaseio.com/"
})

#importing student images
folderPath = 'Images'
PathList = os.listdir(folderPath)
print(PathList)
imgList = []
StudentIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    StudentIds.append(os.path.splitext(path)[0])




   # print(path)
    #print(os.path.splitext(path)[0])
print(StudentIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

print("encoding started...")
encodeListKnown = findEncodings(imgList)
#print(encodeListKnown)
encodeListKnownwithids=[encodeListKnown,StudentIds]
print("encoding end.....")

file= open("encodeFile.p","wb")
pickle.dump(encodeListKnownwithids,file)
file.close()
print("file saved")