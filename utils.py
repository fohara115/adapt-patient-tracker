import pyrealsense2 as rs
import numpy as np
import cv2
import os.path
import sys
import getopt
import time
import Jetson.GPIO as GPIO
from datetime import datetime


def lcd_shutdown_msg(lcd_monitor):
    lcd_monitor.write("Powering Off...\n".encode('utf-8'))
    time.sleep(5)


def lcd_boot_msg(lcd_monitor):
    time.sleep(2)
    lcd_monitor.write("Powering On...\n".encode('utf-8'))


def get_features_v4(orb, img, bbox):
    roi = cut_bbox(img, bbox)
    gimg = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    kp1 = orb.detect(gimg, None)

    m1 = np.full(roi.shape[0], False, dtype=bool)
    m1[roi.shape[0]//4:3*roi.shape[0]//4] = True

    
    return np.array([
        len(kp1),
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],       
        np.round(np.mean(roi[m1,:,0])),
        np.round(np.mean(roi[m1,:,1])),
        np.round(np.mean(roi[m1,:,2])),
        #np.round(np.std(roi[m1,:,0])),
        #np.round(np.std(roi[m1,:,1])),
        #np.round(np.std(roi[m1,:,2])),
        np.sum(gimg > 1e-2) / (bbox[2]*bbox[3])
    ])


def get_features_v3(kp, roi, bbox):
  
    return np.array([
        len(kp),
        len(kp)*len(kp),
        bbox[2],
        bbox[3],
        bbox[2]*bbox[3],
        bbox[1],
        np.round(np.mean(roi[:,:,0])),
        np.round(np.mean(roi[:,:,1])),
        np.round(np.mean(roi[:,:,2]))
    ])


def get_features_v2(orb, img, bbox):
    roi = cut_bbox(img, bbox)
    kp, des = orb.detectAndCompute(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), None)
    
    return np.array([
        len(kp),
        len(kp)*len(kp),
        bbox[2],
        bbox[3],
        bbox[2]*bbox[3],
        bbox[1],
        np.round(np.mean(roi[:,:,0])),
        np.round(np.mean(roi[:,:,1])),
        np.round(np.mean(roi[:,:,2]))
    ])


def get_features(surf, img, bbox):
    roi = cut_bbox(img, bbox)
    kp = surf.detect(roi, None)
    
    return np.array([
        len(kp),
        len(kp)*len(kp),
        bbox[2],
        bbox[3],
        bbox[2]*bbox[3],
        bbox[1],
        np.round(np.mean(roi[:,:,0])),
        np.round(np.mean(roi[:,:,1])),
        np.round(np.mean(roi[:,:,2])),
        np.round(np.max(roi[:,:,0])),
        np.round(np.max(roi[:,:,1])),
        np.round(np.max(roi[:,:,2]))
    ])


def cut_bbox(img, bbox):
    return img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]


def calculate_ang(p1, p2, img_w, conversion):
    return ((p1[0] + p2[0]//2) - img_w//2) * conversion / img_w


def calculate_dist_from_roi(dep_img, p1, p2, bbmin, qmin):
    roi = dep_img[p1[0]:p2[0], p1[1]:p2[1]]
    filt = roi[roi > bbmin]
    if filt.size > 0:
        return np.percentile(filt, qmin)
    else:
        return None


def update_lcd_display(lcd_monitor, tracker_init, d, a, missing, ui_state, fps):
    # LINE1
    if ui_state == 0:
        s = 1
    elif missing:
        s = 3
    else:
        s = 2
    lcd_monitor.write(f"ADAPT ({s},{int(tracker_init)}) {int(np.round(fps, 0))}Hz\n".encode('utf-8'))
    
    if tracker_init and d and a:
        d_value = int(np.round(d*100, 0)) if d else d
        a_value = int(np.round(a,0)) if a else a
    else:
        d_value = ' ---'
        a_value = ' ---'

    msg = "d:"+f"{d_value}".rjust(4) + "cm a:" + f"{a_value}".rjust(3) + "\n"
    lcd_monitor.write(msg.encode('utf-8'))


def full_height_box(bbox, img_h, img_w, width=200):
    center_x = int(bbox[0] + bbox[2]//2)
    p1 = (center_x - width//2, 0)
    p2 = (center_x + width//2, img_h)

    return p1, p2    


def send_d_stop(d_port):
    for _ in range(5):
        d_port.write(f"1,None,4,None\n".encode('utf-8'))


def send_a_stop(a_port):
    for _ in range(5):
        a_port.write(f"2,None,4,None\n".encode('utf-8'))


def check_estop(lcd_monitor):
     if lcd_monitor.in_waiting > 0:
         return True
     else: 
         return False


def read_gpio_state(B0_PIN, B1_PIN):
    b0 = GPIO.input(B0_PIN)
    b1 = GPIO.input(B1_PIN)

    if b1==0 and b0==0:
        return 0
    elif b1==0 and b0==1:
        return 1
    elif b1==1 and b0==0:
        return 2
    elif b1==1 and b0==1:
        return 3
    else:
        return None


def update_lcd_board_state(lcd_monitor, ui_state):
    if ui_state == 0:
        lcd_monitor.write(f"S1\n".encode('utf-8'))
    else:
        lcd_monitor.write(f"S2\n".encode('utf-8'))


def send_d_signals(d_port, d, ui_state, tracker_init, missing):

    d_value = np.round(d*100,1) if d else d

    if (not tracker_init):
        state = 1
    elif (ui_state == 0):
        state = 1
    elif (missing):
        state = 3
    else:
        state = 2
    d_port.write(f"1,{d_value},{state},{ui_state}\n".encode('utf-8'))


def send_a_signals(a_port, a, d, ui_state, tracker_init, missing):
    if d is not None:
        if d < 1:
            a = 0    

    a_value = np.round(a,1) if a else a
    if (not tracker_init):
        state = 1
    elif (ui_state == 0):
        state = 1
    elif (missing):
        state = 3
    else:
        state = 2
    a_port.write(f"2,{a_value},{state}\n".encode('utf-8'))


def send_adam_signals(d_port, a_port, d, a, ui_state, tracker_init, missing):
    'Send two ports to adam'
    
    d_value = np.round(d,3) if d else d
    a_value = np.round(a,3) if a else a
    if (not tracker_init):
        state = 1
    elif (ui_state == 0):
        state = 1
    elif (missing):
        state = 3
    else:
        state = 2
    d_port.write(f"1,{d_value},{state},{ui_state}\n".encode('utf-8'))
    a_port.write(f"2,{a_value},{state},{ui_state}\n".encode('utf-8'))


def update_fps(fps, tic):
    toc = time.time()
    curr_fps = 1.0 / (toc - tic)
    fps = curr_fps if fps == 0.0 else (fps*0.5 + curr_fps*0.5)
    return fps, toc

def process_cli_args_wstate(iroot, oroot, default, live, state):
    'process directories and filenames for io'

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'e')

    if (len(args)==0):
        input_path = iroot+default
        output_path = oroot+'default_'+default
    elif (len(args)==1):
        input_path = iroot+args[0]
        output_path = oroot+args[0]
    elif (len(args)>1):
        input_path = iroot+args[0]
        output_path = oroot+args[1]+args[0]

    if live:
        now = datetime.now()
        output_path = oroot+str(state)+'_'+now.strftime("%Y%m%d_%H%M%S")+'.txt'

    return input_path, output_path


def process_cli_args(iroot, oroot, default, live):
    'process directories and filenames for io'

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'e')

    if (len(args)==0):
        input_path = iroot+default
        output_path = oroot+'default_'+default
    elif (len(args)==1):
        input_path = iroot+args[0]
        output_path = oroot+args[0]
    elif (len(args)>1):
        input_path = iroot+args[0]
        output_path = oroot+args[1]+args[0]

    if live:
        now = datetime.now()
        output_path = oroot+now.strftime("%Y%m%d_%H%M%S")+'.txt'

    return input_path, output_path


def get_center_distance(depth_image, img_height=480, img_width=640, roi_height=256, roi_width=256, percentile=10):
    '''Return center region of the frame to detect patient seated mode'''
    if depth_image.size == 0:
        return None

    x_range = [(img_width//2 - roi_width//2), (img_width//2 + roi_width//2)]
    y_range = [(img_height//2 - roi_height//2), (img_height//2 + roi_height//2)]
    vals = depth_image[x_range[0]:x_range[1], y_range[0]:y_range[1]][depth_image[x_range[0]:x_range[1], y_range[0]:y_range[1]] > 0]

    if (vals.size > 0):
        return np.percentile(vals, percentile)
    else:
        return 0


def format_frames(align, frames, depth_scale):
    '''Create cleaned np arrays from pipeline frames and align'''

    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return True, None, None
    
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    if depth_image.size == 0 or color_image.size == 0:
        return True, None, None

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    depth_image = depth_image * depth_scale

    return False, color_image, depth_image


def depth_masking(depth_image, clip_dist, image_height=480, image_width=640, num_channels=3):
    '''480x640x3 Mask of 1's and 0's from depth frame'''
    mask = np.where((depth_image < clip_dist) & (depth_image > 1e-6), (1), np.zeros((image_height, image_width), np.uint8))
    
    return np.dstack((mask,mask,mask))


def person_masking(boxes, image_height=480, image_width=640, num_channels=3):
    '''480x640x3 Mask of 1's and 0's from classification boxes'''
    mask = np.zeros((image_height, image_width, num_channels), np.uint8)
    for xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (1, 1, 1), -1)

    return mask


def person_masking_depr(boxes, person_mask_queue, image_height=480, image_width=640, num_channels=3):
    '''480x640x3 Mask of 1's and 0's from classification boxes'''
    mask = np.zeros((image_height, image_width, num_channels), np.uint8)
    for xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (1, 1, 1), -1)
    person_mask_queue = np.append(person_mask_queue[:, :, :, 0:person_mask_queue.shape[3]-1], mask[:,:,:,np.newaxis], axis=3)
    full_mask = np.sum(person_mask_queue, axis=3) > 0
    return full_mask
    

def get_depth_scale(profile):
    '''Simple helper to get depth sensor scale from realsense'''
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return depth_scale


def load_bag_file(path):
    """
    PURPOSE: Create rs pipeline with a proper config from a .bag file

    INPUTS:
    * path: string containing the path and filename for .bag file of choice
    """

    if os.path.splitext(path)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()
    
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, path)

    FPS = 30
    config.enable_all_streams()

    return pipeline, config


def load_live_stream():
    """
    PURPOSE: Create rs pipeline with proper config for a live camera feed

    INPUTS: None
    """

    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    FPS = 30
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
    config.enable_all_streams()

    return pipeline, config


def display_pipeline(pipeline, config):
    """
    OVERVIEW: From a stream, simply plot onto computer
    """

    pipeline.start(config)

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # If depth and color resolutions are different, resize color image to match depth image for display
            # if depth_colormap_dim != color_colormap_dim:
            #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            #     images = np.hstack((resized_color_image, depth_colormap))
            # else:
            #     images = np.hstack((color_image, depth_colormap))
            # resized_color_image = cv2.resize(color_image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            # resized_depth_colormap = cv2.resize(depth_colormap, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            # images = np.hstack((resized_color_image, resized_depth_colormap))
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    finally:
        pipeline.stop()


def align_and_display(pipeline, config):

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 4 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))

            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

