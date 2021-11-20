import cv2
import numpy as np

photo = 'Photo1.jpg'

img = cv2.imread(photo)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
boxes = cascade.detectMultiScale(gray, 1.3, 10)

eyes = []
if len(boxes) == 2:
    for box in boxes:
        x, y, w, h = box
        eye = img[y:y + h, x:x + w]
        eyes.append(eye)

photo_name = photo.split('.jpg')[0]
cv2.imwrite(photo_name + '-left_eye.jpg', eyes[0])
cv2.imwrite(photo_name + '-right_eye.jpg', eyes[1])
