import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

# Define the input folder containing PNG frames and the input CSV file
frame_folder = 'frames/20231201_152420/'  # Replace with your frame folder path
csv_file = 'proc_labels/20231201_152420.csv'     # Replace with your CSV file path

# Output video parameters
output_video_path = 'label_20231201_152420.mp4'
frame_width, frame_height = 640, 480  # Adjust as needed
frame_rate = 30

# Function to read coordinates from CSV file
def read_coordinates_from_csv(file_path):
    coordinates = {}

    df = pd.read_csv(file_path)
    # with open(file_path, 'r') as f:
    #     for line in f:
    #         frame_idx, x, y = map(int, line.strip().split(','))
    #         coordinates[frame_idx] = (x, y)
    return df

# Function to add a circle to a frame
def add_circle_to_frame(frame, center, radius=25):
    return cv2.circle(frame.copy(), center, radius, (0, 0, 255), -1)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

# Read coordinates from CSV
coordinates = read_coordinates_from_csv(csv_file)

# Process frames and create the video
for frame_idx in tqdm(range(len(os.listdir(frame_folder)))):
    frame_path = os.path.join(frame_folder, os.listdir(frame_folder)[frame_idx])
    frame = cv2.imread(frame_path)
    if frame is None:
        continue
    
    if frame_idx in coordinates['idx']:
        circle_center = (int(coordinates['x_coord'][frame_idx]), int(coordinates['y_coord'][frame_idx]))
        frame = add_circle_to_frame(frame, circle_center)
    
    frame = cv2.resize(frame, (frame_width, frame_height))
    video_out.write(frame)

# Release video writer
video_out.release()

print(f"Video saved to '{output_video_path}'")
