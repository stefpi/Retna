from ctypes import resize
import os
import cv2
import numpy as np

# root = './data/'
# filepaths = os.listdir(root)
# X, Y = [], []
# for filepath in filepaths:
#     img = cv2.imread(root + filepath)
#     resized = cv2.resize(img, (44, 12))
#     cv2.imwrite(root + filepath, resized)

cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

def normalize(x):
  minn, maxx = x.min(), x.max()
  return (x - minn) / (maxx - minn)

def scan(frame, image_size=(32, 32)):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  boxes = cascade.detectMultiScale(gray, 1.3, 10)
  if len(boxes) == 2:
    eyes = []
    for box in boxes:
      x, y, w, h = box
      eye = frame[y:y + h, x:x + w]
      eye = cv2.resize(eye, image_size)
      eye = normalize(eye)
      eye = eye[10:-10, 5:-5]
      eyes.append(eye)
    return (np.hstack(eyes) * 255).astype(np.uint8)
  else:
    return None

root = './Ben/'
cropped = './Bencropped/'
filepaths = os.listdir(root)
X, Y = [], []
for filepath in filepaths:
    img = cv2.imread(root + filepath)
    cropped_img = scan(img)
    if cropped_img is not None:
        cv2.imwrite(cropped + filepath, cropped_img)
        print(filepath + " cropped")
    else:
        print("no eyes detected")
