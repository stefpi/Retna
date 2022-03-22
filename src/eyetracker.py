import numpy as np
import os
import cv2
import pyautogui
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)

def normalize(x):
  minn, maxx = x.min(), x.max()
  return (x - minn) / (maxx - minn)

def scan(image_size=(32, 32)):
  _, frame = video_capture.read()
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

# Note that there are actually 2560x1440 pixels on my screen
# I am simply recording one less, so that when we divide by these
# numbers, we will normalize between 0 and 1. Note that mouse
# coordinates are reported starting at (0, 0), not (1, 1)
width, height = 2559, 1599

root = './newcropped/'
filepaths = os.listdir(root)
X, Y = [], []
for filepath in filepaths:
  print('loading ' + filepath + ' ...')
  filename = filepath[0:filepath.rfind(".")]
  x, y = filename.split('-')
  #print(x, y)
  x = float(x) / width
  y = float(y) / height
  z = cv2.imread(root + filepath)
  X.append(z)
  Y.append([x, y])
X = np.array(X) / 255.0
#X = np.array(X).astype(np.float32)
Y = np.array(Y)
#Y = np.array(Y).astype(np.float32)

model = Sequential()
model.add(Conv2D(32, 3, 2, activation = 'relu', input_shape = (12, 44, 3)))
model.add(Conv2D(64, 2, 2, activation = 'relu'))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(optimizer = "adam", loss = "mean_squared_error")
model.summary()

epochs = 200
for epoch in range(epochs):
  model.fit(X, Y, batch_size = 32)

print("done")
print("scanning starting")

old_x, old_y = 0.1, 0.1
threshold = 0.3
while True:
  eyes = scan()
  print("scanned")
  if not eyes is None:
      print("found eyes")
      eyes = np.expand_dims(eyes / 255.0, axis = 0)
      x, y = model.predict(eyes)[0]
      print(x, y)
      if (x - old_x)/old_x > threshold or (y - old_y)/old_y > threshold:
        pyautogui.moveTo(x * width, y * height)
      old_x, old_y = x, y