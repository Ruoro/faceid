import cv2
import os
import pickle
import face_recognition
import numpy as np


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('resources/backg.png')
folderModePath = 'resources/modes/'
mode_path_list = os.listdir(folderModePath)
img_mode_list = []

# importing mode images into a list
for x_path in mode_path_list:
    img_mode_list.append(cv2.imread(os.path.join(folderModePath, x_path)))

# load the encoding file
print("Loading Encoding File ...")
file = open("encodingFile.p", "rb")
encodeKnownListWithIds = pickle.load(file)
file.close()
print("Encoding File Loaded")

# print(encodeKnownListWithIds)
encodeKnownList, studentIds = zip(*encodeKnownListWithIds)
# print('studentIds', studentIds)
# print('encodeKnownList', encodeKnownList)
# print(len(encodeKnownList))
encodeKnownList =  encodeKnownList[1]

encodeKnownList = np.array(encodeKnownList)
print('encodeKnownList', encodeKnownList)


while True:
    success, img = cap.read()
    # print('shape of image',img.shape)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25 )
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    
    encodeCurFrame = np.array(encodeCurFrame)
    encodeCurFrame = encodeCurFrame.astype(np.float64)


    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44+630, 808:808+414] = img_mode_list[3]



    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeKnownList, encodeFace)
        faceDis = face_recognition.face_distance(encodeKnownList, encodeFace)
        print ('matches', matches)
        print ('faceDis', faceDis)

        matchIndex = np.argmin(faceDis)
        print('matchIndex', matchIndex)

    cv2.imshow('Face Attendance', imgBackground)
    cv2.waitKey(2)