from __future__ import print_function
import cv2 as cv
import argparse
import os

path = 'Retna\src\Data'
path2 = 'Retna\src\Data_Cropped'
files = os.listdir(path)

def detectAndDisplay(frame, path):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame, (x, y), (x+w, y+h), 
                  (0, 0, 255), 2)
        frame = frame[y:y+h,x:x+w]
    cv.imshow('Capture - Face detection', frame)
    cv.imwrite(path, frame)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='haarcascade_frontalface_althaarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile("Retna\src\haarcascade_frontalface_alt.xml")):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile("Retna\src\haarcascade_eye_tree_eyeglasses.xml")):
    print('--(!)Error loading eyes cascade')
    exit(0)

for file in files:
    file_path = path + "\\"+ file
    file_path2 = path2 + "\\"+ file
    detectAndDisplay(cv.imread(file_path), file_path2)


