import cv2
import face_recognition
import os
import pickle

#Importing Student Images
folderPath = "images"
pathList = os.listdir(folderPath)
imgList = []
student_Ids = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    student_Ids.append(os.path.splitext(path)[0])
    # print(student_Ids)
    print(len(imgList))

def find_encodings(imgList):
    encode_list = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        encode_list.append(encodings)

    return encode_list

print("Encoding Started ...")
encodeListKnown = find_encodings(imgList)

encodeListKnownWithIds = list(zip(encodeListKnown, student_Ids))

print("Encoding Complete")

file = open("encodingFile.p", "wb")
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saving Completed")
print (encodeListKnownWithIds)







