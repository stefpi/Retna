import sys
import cv2
from pynput.mouse import Listener
import os
import shutil

# define a video capture object
vid = cv2.VideoCapture(0)
# Request input for where images will be stored
root = 'Retna\src\Data\\'
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

def snap_picture():
     # .read returns a tuple (whether read was successful, image)
    _, frame = vid.read()
    return frame
        
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
            filename = root + "{}-{}.jpeg".format(x,y)
            print(filename)
            cv2.imwrite(filename, eye)
        else:
            print("Picture not taken.")
    
    
    
with Listener(on_click = on_click) as listener:
    listener.join()
