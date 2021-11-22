from __future__ import print_function
import cv2 as cv
import argparse
import os
import numpy as np

path = 'Retna\src\Data' # Folder of the captured data
path2 = 'Retna\src\Data_Cropped' # Output folder - faces 
path3 = 'Retna\src\Data_Eyes' # Output folder - eyes
files = os.listdir(path)

def detectFace(frame, path):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame, (x, y), (x+w, y+h), 
                  (0, 0, 255), 2)
        frame = frame[y:y+h,x:x+w]
    cv.imwrite(path, frame)

def coordinateCheck(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame, (x, y), (x+w, y+h), 
                  (0, 0, 0), 2)
        frame = frame[y:y+h,x:x+w]
        print(x, y, w, h)

def detectEyes(frame, path):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect eyes
    boxes = eyes_cascade.detectMultiScale(frame_gray)
    
    total_width = 0
    total_height = 0
    
    if len(boxes) == 2:
        for box in boxes:
            x, y, w, h = box
            # Determine the size of the final image
            total_width += w
            if h > total_height:
                total_height = h

        transparent_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        w_start=boxes[0][2] # This is the x-value from which the second box renders onto the image

        for i in range(2):
            x, y, w, h = boxes[i]
            transparent_img[0:h, i*w_start:i*w_start+w] = frame[y:y + h, x:x + w]

        cv.imwrite(path, transparent_img)

    elif len(boxes) == 1:
        for (x,y,w,h) in boxes:
            frame = cv.rectangle(frame, (x, y), (x+w, y+h), 
                    (0, 0, 255), 2)
            frame = frame[y:y+h,x:x+w]
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

# Load the cascades
if not face_cascade.load(cv.samples.findFile("Retna\src\haarcascade_frontalface_alt.xml")):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile("Retna\src\haarcascade_eye_tree_eyeglasses.xml")):
    print('--(!)Error loading eyes cascade')
    exit(0)

for file in files:
    file_path = path + "\\"+ file
    file_path2 = path2 + "\\"+ file
    file_path3 = path3 + "\\"+ file
    detectEyes(cv.imread(file_path), file_path3)
    #detectFace(cv.imread(file_path), file_path2)
    #coordinateCheck(cv.imread('Retna/src/Data/686-72.jpeg'))


