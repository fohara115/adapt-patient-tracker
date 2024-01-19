import os
import cv2
import tkinter as tk
from tkinter import filedialog

folder_path = "C:/Users/finno/Workspace/adapt-patient-tracker/frames/20231201_152420/"
file_path = "labels/20231201_152420_label1.txt"

def label_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates (x={x}, y={y}), {index}")
        with open(file_path, "a") as file:
            # Add new lines to the file
            file.write(str(index)+','+str(x)+','+str(y)+'\n')

if not folder_path:
    exit()

# Loop through PNG images in the folder
index = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".png") and filename.startswith('_Color_'):
        if index % 2 == 0:
            index = index + 1
            continue

        image_path = os.path.join(folder_path, filename)
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {filename}")
            continue
        
        # Create a window for labeling
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", image)

        # Set up the mouse event callback
        cv2.setMouseCallback("Image", label_pixel)
        
        cv2.waitKey(0)

        index = index + 1

        cv2.destroyAllWindows()

print("Labeling process completed.")
