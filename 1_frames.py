import pyrealsense2 as rs
import numpy as np
import cv2
import os

from load import load_bag_file, load_live_stream



# ----- LOAD CONFIG -----

date = "20231201_152420"
filename = 'D:/bme/data/20231201_152420.bag'
output_frames_dir = 'D:/bme/labels/frames/20231201_152420/'
clip_limit = 4



# ----- VIDEO & TRACKER SETUP -----

# Setup video origin
pipeline, config = load_bag_file(filename)

# Setup video stream w/ alignment
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance = clip_limit / depth_scale
align_to = rs.stream.color
align = rs.align(align_to)

if not os.path.exists(output_frames_dir):
    os.makedirs(output_frames_dir)




# ----- MAIN LOOP -----

t_prev = 0
while True:
    frames = pipeline.wait_for_frames()

    t = frames.get_timestamp()
    if (t < t_prev):
        break
    t_prev = t

    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        print('Skipping problematic frame...')
        continue
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    t = frames.get_timestamp()

    cv2.imwrite(f"{output_frames_dir}frame__{date}__{int(t)}.png", color_image)

pipeline.stop()
