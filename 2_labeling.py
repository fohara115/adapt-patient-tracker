import os
import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

# Setup

date = '20240317_153447'
input_frames_dir = 'D:/bme/frames/20240317_153447/'
output_dir = f'D:/bme/labels/coords/tslab_{date}.txt' 

with open(output_dir, 'w') as o:
    o.write(f"LABELS for {date}\n")


def label_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates (x={x}, y={y}), {index}")
        with open(output_dir, "a") as file:
            # Add new lines to the file
            file.write(str(ts)+','+str(x)+','+str(y)+',None\n')




# Run
skip_margin = 10

index = 0
for filename in tqdm(os.listdir(input_frames_dir)):
    if filename.endswith(".png"):
        if index % skip_margin == 0:
            index = index + 1
            continue

        image_path = os.path.join(input_frames_dir, filename)
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {filename}")
            continue
        
        # Create a window for labeling
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", image)

        # Set up the mouse event callback
        last_piece = filename.split('__')[-1]
        ts = last_piece.split('.')[0]
        cv2.setMouseCallback("Image", label_pixel)
        
        cv2.waitKey(0)

        index = index + 1

        cv2.destroyAllWindows()

print("Labeling process completed.")
