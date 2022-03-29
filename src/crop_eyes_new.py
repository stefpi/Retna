import numpy as np
import cv2
import os
from os import listdir

folder_dir = r"C:\Users\danie\Desktop\pies\Daniel"
output_path = r"C:\Users\danie\Desktop\pies\NewHope"

for image in os.listdir(folder_dir):
    print(image)
    photo = image
    
    # Read the input image
    img = cv2.imread(photo)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, 1.3, 10)

    eyes = []
    if len(boxes) == 2:
        x1, y1, w1, h1 = boxes[0]
        x2, y2, w2, h2 = boxes[1]
        if x1 < x2:
            eyeL = img[y1:y1 + h1, x1:x1 + w1]
            eyeR = img[y2:y2 + h1, x2:x2 + w1]
        if x1 > x2:
            eyeL = img[y2:y2 + h1, x2:x2 + w1]
            eyeR = img[y1:y1 + h1, x1:x1 + w1]
    vis = np.concatenate((eyeL, eyeR), axis=1)
    name = photo.strip('.jpeg') + 'New.jpeg'

    cv2.imwrite(os.path.join(output_path , name), vis)
