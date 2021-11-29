from __future__ import print_function
import sys
import cv2 as cv
from pynput.mouse import Listener
import shutil
import argparse
import os
import numpy as np

# define a video capture object
vid = cv.VideoCapture(0)
# Request input for where images will be stored
root = r'C:\Users\danie\Pictures\Retna'
eyes_path = r'C:\Users\danie\Pictures\RetnaEyes'
# Check if input is a directory that already exists
if os.path.isdir(root):
    response = ""
    # While response is not y or n, keep requesting for input
    while not (response in ["y", "n"]):
        response = input("Would you like to overwrite directory?[y/n]")
        if (response == "y"):
            shutil.rmtree(root)
            print("Directory overwritten.")
        else:
            exit()
os.mkdir(root)

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
if not face_cascade.load(cv.samples.findFile(r"C:\Users\danie\Documents\GitHub\Retna\src\haarcascade_frontalface_alt.xml")):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(r"C:\Users\danie\Documents\GitHub\Retna\src\haarcascade_eye_tree_eyeglasses.xml")):
    print('--(!)Error loading eyes cascade')
    exit(0)

def snap_picture():
     # .read returns a tuple (whether read was successful, image)
    _, frame = vid.read()
    return frame

def detectEyes(frame, path):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect eyes
    boxes = eyes_cascade.detectMultiScale(frame_gray)
    
    max_height = 0 # Ideally we want to upscale the smaller frame so we dont lose information
    total_width=0

    if len(boxes) == 2:
        for i in range(2):
            h = boxes[i][3]
            w = boxes[i][2]
            # Determine the size of the final image
            total_width+=w
            if h > max_height:
                max_height = h

        final = np.zeros((max_height,total_width,3), np.uint8) # initializes final image
        w_start=boxes[0][2] # This is the x-value from which the second box renders onto the image

        for i in range(2):
            x, y, w, h = boxes[i]
            delta = 0
            if h < max_height:
                delta = round((max_height-h)/2)
            final[0: max_height, i*w_start:i*w_start+w] = frame[(y-delta):(y+max_height-delta),x:x+w] # This will fail if the eye is at the corner of screen
            
        cv.imwrite(path, final)

    elif len(boxes) == 1:
        for (x,y,w,h) in boxes:
            frame = frame[y:y+h,x:x+w]
        cv.imwrite(path, frame)

def on_click(x, y, button, pressed):
    """
    Args:
    x: the x-coordinate of the mouse
    y: the y-coordinate of the mouse
    button: 1 or 0, depending on right-click or left-click
    pressed: 1 or 0, whether the mouse was pressed or released
    """
    if pressed:
        print (x, y)
        eye = snap_picture()
        # If eye successfully took picture, save it
        if not (eye is None):
            filename = "{}-{}.jpeg".format(x,y)
            print(filename)
            cv.imwrite(os.path.join(root, filename), eye)
            tempFile = root + "\\"+ filename
            detectEyes(cv.imread(tempFile), eyes_path + "\\"+ filename)
        else:
            print("Picture not taken.")

with Listener(on_click = on_click) as listener:
    listener.join()
