import numpy as np
import os
import cv2
import pyautogui
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

cascade = cv2.CascadeClassifier("./src/haarcascade_eye_tree_eyeglasses.xml")
video_capture = cv2.VideoCapture(0)
# Change this to ur resolution, but minus 1 pixel for both dimension
width, height = 1919, 1199


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


def train_model(root):
  filepaths = os.listdir(root)
  X, Y = [], []
  for filepath in filepaths:
    print('loading ' + filepath + ' ...')
    filename = filepath[0:filepath.rfind(".")]
    xy = filename.split('-')
    # Fix some issue
    if len(xy) != 2:
      continue
    x, y = xy
    #print(x, y)
    x = float(x) / width
    y = float(y) / height
    z = cv2.imread(root + filepath)
    rz = cv2.resize(z, (44, 12))
    X.append(rz)
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

  model.save("./Saved Models/model0")
  print("done training")


def mean(arr):
  mean = 0
  for i in arr:
    mean += i

  return mean/len(arr)


def scroller(model):
  """
  Allows for scrolling with the eye
  """
  counter = 0
  state = "neutral"
  prev_scans = []
  while True:
    eyes = scan()
    if not eyes is None:
      # Prolongs scroll, not working properly
      """if state is "up" or state is "down":
        eyes = np.expand_dims(eyes / 255.0, axis = 0)
        x, y = model.predict(eyes)[0]
        print(y)
        # Scroll up
        if y < 0.3 and state is "up":
          pyautogui.scroll(10)

        elif y < height * 0.3 and state is "down":
          state = "neutral"

        # Scroll Down 
        elif y > 0.7 and state is "down":
          pyautogui.scroll(-10)

        elif y > 0.7 and state is "down":
          state = "neutral"

        counter = 0
        prev_scans = []
        continue"""

      if counter == 4:
        # Scroll up
        if mean(prev_scans) > 0.7:
          pyautogui.scroll(10)
          state = "up"
        # Scroll Down 
        elif mean(prev_scans) < 0.5:
          pyautogui.scroll(-10)
          state = "down"
        counter = 0
        prev_scans = []
      eyes = np.expand_dims(eyes / 255.0, axis = 0)
      x, y = model.predict(eyes)[0]
      print(y)
      # We just need the vertical coordinates
      prev_scans.append(y)
      counter += 1

if __name__ == "__main__":
  model = load_model("./Saved Models/model0")
  model.summary()

  print("scanning starting")

  scroller(model)
  """old_x, old_y = 0.1, 0.1
  threshold = 0.3
  while True:
    eyes = scan()
    if not eyes is None:
        eyes = np.expand_dims(eyes / 255.0, axis = 0)
        x, y = model.predict(eyes)[0]
        print(x, y)
        pyautogui.moveTo(x * width, y * height)"""