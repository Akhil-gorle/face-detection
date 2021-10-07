import cv2
import csv
import numpy as np
import face_recognition
import os
from datetime import *
import winsound
import pyttsx3


frequency = 1000
NewVoiceRate = 100
duration = 1000  # ms


engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', NewVoiceRate)
#engine.say('hi hello welcom')
engine.runAndWait()


path = 'imagestrain'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def createFile():
    today = date.today()
    file = os.listdir()
    if 'mycsv{}{}{}.csv'.format(today.day, today.month, today.year) in file:
        return
    else:
        with open('mycsv{}{}{}.csv'.format(today.day, today.month, today.year), 'w', newline='') as f:
            fieldnames = ['Name', 'Time']
            thewriter = csv.DictWriter(f, fieldnames=fieldnames)
            thewriter.writeheader()


def markAttendance(name):
    today = date.today()
    createFile()

    with open('mycsv{}{}{}.csv'.format(today.day, today.month,today.year),'r+', newline='') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            engine.say('hi {} good morning'.format(name))
            engine.runAndWait()


encodeListKnown = findencodings(images)
print("encoding complete")

cap = cv2.VideoCapture(0)
img_counter = 0

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    k = cv2.waitKey(1)

    if k == 32:
        a = input('please enter your name: ')
        cv2.imwrite('imagestrain/{}.jpg'.format(a), img)

    # elif k == 27:
    #     print('Escape hit, closing the app')
    #     break

    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodesCurrentFrame = face_recognition.face_encodings(imgS,faceCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("distances : ", faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x1, y2, x2 = faceLoc
            y1, x1, y2, x2 = y1*4, x1*4, y2*4, x2*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x2, y2), (x1, y2+35), (100, 200, 210), cv2.FILLED)
            cv2.putText(img, name, (x2+6, y2+30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            #cv2.circle(img, (x2, y2), 5,(255,0,0),2)
            markAttendance(name)

        else:
            print("error")
            now = datetime.now()
            dtString = now.strftime('%H%M%S')
            cv2.imwrite('error_images/error_img_{}.png'.format(dtString), img)
            img_counter = img_counter + 1
            winsound.Beep(frequency, duration)






    cv2.imshow('Webcam', img)
    # cv2.waitKey(1)
