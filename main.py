import pyrealsense2 as rs
import numpy as np
import cv2
import yaml

from load import load_bag_file, load_live_stream



# ----- LOAD CONFIG -----

cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.CLoader)
live_input = cfg['input']['live']
if (not live_input):
    filename = cfg['input']['root'] + cfg['input'][cfg['input']['select']]

clip_limit = cfg['processing']['clip_limit']
replace_color = cfg['processing']['clip_replace']
image_height = cfg['processing']['img_height']
image_width = cfg['processing']['img_width']
tracker_type = cfg['tracker']['type']
bbox_height = cfg['tracker']['bbox_height']
bbox_width = cfg['tracker']['bbox_width']
bbox_qmin = cfg['tracker']['bbox_qmin']
distance_trigger = cfg['tracker']['distance_trigger']
disp = cfg['runtime']['display_window']
print_dist = cfg['runtime']['print_distance']
print_coord = cfg['runtime']['print_coord']
write_output = cfg['runtime']['write_output']
output_dir = cfg['runtime']['output_root']+cfg['runtime']['output_filename']



# ----- VIDEO & TRACKER SETUP -----

# Setup video origin
if (not live_input) and filename: 
    pipeline, config = load_bag_file(filename)
else: 
    pipeline, config = load_live_stream()

# Setup video stream w/ alignment
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance = clip_limit / depth_scale
align_to = rs.stream.color
align = rs.align(align_to)
t_prev = 0

# Setup tracker
init_bbox = (image_width//2 - (bbox_width//2), image_height//2 - (bbox_height//2), bbox_height, bbox_width)
tracker_init = False
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
else:
    raise Exception('Tracker type us bit supported.')

# Setup output
if write_output:
    with open(output_dir, 'w') as o:
        o.write(f"OUTPUT for LIVE: {live_input}   INPUT: {filename}\n")



# ----- MAIN LOOP -----

while True:
    # Load valid depth and colour frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        print('Skipping problematic frame...')
        continue
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    t = frames.get_timestamp()


    # Remove background and render images
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), replace_color, color_image)
    if disp:
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

    # Tracker updating
    center_dist = depth_image[image_height//2, image_width//2] * depth_scale
    if (center_dist > distance_trigger) and (not tracker_init): # Switch tracker on
        ret = tracker.init(bg_removed, init_bbox)
        tracker_init = True 
    elif (center_dist < distance_trigger) and (tracker_init) and (center_dist > 1e-6): # Switch tracker off
        tracker_init = False
        print('too close! turning off')

    if tracker_init:
        ret, bbox = tracker.update(bg_removed)
        if ret and disp:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(images, p1, p2, (255,0,0), 2, 1)

        # Calculate signals of interest
        if ret:
            bbox_min_dist = np.percentile(depth_image[int(bbox[0]):int(bbox[0]+bbox[2]), int(bbox[1]):int(bbox[1]+bbox[3])] * depth_scale, bbox_qmin)
        if print_dist:
            print(bbox_min_dist)
        if print_coord:
            print(f"Centre: ({int(bbox[0] + bbox[2]//2)}, {int(bbox[1] + bbox[3]//2)})")
    
    if disp:
        #cv2.namedWindow('ADAPT Patient Tracker', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense Sensors', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

    if write_output:
        with open(output_dir, "a") as f:
            if (tracker_init and ret):
                print(f"{t},{int(bbox[0] + bbox[2]//2)},{int(bbox[1] + bbox[3]//2)},{bbox_min_dist}", file=f)
            else:
                print(f"{t},None,None,{center_dist}", file=f)

    # Check for end of recorded tape 
    if (not live_input):
        if (t < t_prev):
            break
        t_prev = t

pipeline.stop()