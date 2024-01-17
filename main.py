import pyrealsense2 as rs
import numpy as np
import cv2
from utils import display_pipeline, align_and_display
from load import load_bag_file, load_live_stream


# Get video stream
filename = "data/20231201_152820.bag"
pipeline, config = load_bag_file(filename)


# Create the tracker
tracker = cv2.TrackerMIL_create() 


# Alignment Setup
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

clipping_distance_in_meters = 10
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)


# ROI Setup
bbox = (255, 120, 146, 207)
tracker_init = False
# from bbox = cv2.selectROI(bg_removed, False)


# Streaming loop
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        print('Skipping problematic frame...')
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Remove background
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

    # Render images
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((bg_removed, depth_colormap))

    # Update tracker
    if not tracker_init:
        ret = tracker.init(bg_removed, bbox)
        tracker_init = True
    else:
        ret, bbox = tracker.update(bg_removed)

    if ret:
        print(bbox)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(images, p1, p2, (255,0,0), 2, 1)
        

    cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
    cv2.imshow('Align Example', images)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

pipeline.stop()
