import cv2
import numpy as np
import os
import pycuda.autoinit
import pyrealsense2 as rs
import serial
import time
import utils
import yaml

from trt_utils.yolo_classes import get_cls_dict
from trt_utils.camera import add_camera_args, Camera
from trt_utils.display import open_window, set_display, show_fps
from trt_utils.visualization import BBoxVisualization
from trt_utils.yolo_with_plugins import TrtYOLO



# ----- LOAD CONFIG & ARGS -----

cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.CLoader)
LIVE_FEED = cfg['input']['live_feed']
INPUT_ROOT = cfg['input']['root']
DEFAULT_VID = cfg['input']['default']
DISP = cfg['project']['display_window']
ENABLE_LCD = cfg['project']['lcd_monitor']
ENABLE_MOTOR_SIG = cfg['project']['enable_adam_signals']
CLIP_DIST = cfg['video']['clip_limit']
IMAGE_HEIGHT = cfg['video']['img_height']
IMAGE_WIDTH = cfg['video']['img_width']
IMAGE_LFOV_DEG = cfg['video']['img_wide_fov_deg']
DIST_THRESH = cfg['parameters']['seated_trigger']
UI_STATE = cfg['parameters']['ui_state']
TRACKER_TYPE = cfg['tracker']['type']
BBOX_HEIGHT = cfg['tracker']['bbox_height']
BBOX_WIDTH = cfg['tracker']['bbox_width']
BBOX_QMIN = cfg['tracker']['bbox_qmin']
MODEL = cfg['model']['name']
CAT_NUM = cfg['model']['cat_num']
PERSON_CLASS = cfg['model']['person_label']
LETTER_BOX = cfg['model']['letter_box']
CONF_THRESH = cfg['model']['conf_thr']
BAUD = cfg['serial']['baud_rate']
MONITOR_PORT = cfg['serial']['lcd_port']
ADAM_PORT = cfg['serial']['adam_port']
WRITE_OUTPUT = cfg['output']['log_output']
OUTPUT_ROOT = cfg['output']['output_root']

input_dir, output_dir = utils.process_cli_args(iroot=INPUT_ROOT, oroot=OUTPUT_ROOT, default=DEFAULT_VID, live=LIVE_FEED)



# ----- VIDEO SETUP -----

pipeline, config = utils.load_live_stream() if LIVE_FEED else utils.load_bag_file(input_dir)
profile = pipeline.start(config)
depth_scale = utils.get_depth_scale(profile)
align = rs.align(rs.stream.color)
fps = 0



# ----- DETECTOR SETUP -----

cls_dict = get_cls_dict(CAT_NUM)
vis = BBoxVisualization(cls_dict)
trt_yolo = TrtYOLO(MODEL, CAT_NUM, LETTER_BOX)



# ----- TRACKER SETUP -----

if TRACKER_TYPE=='CSRT':
    tracker = cv2.TrackerCSRT_create()
elif TRACKER_TYPE=='KCF':
    tracker = cv2.TrackerKCF_create()
elif TRACKER_TYPE=='MIL':
    tracker = cv2.TrackerMIL_create()
else:
    raise(f'ERROR: Provided tracker type {TRACKER_TYPE} is not supported in this project.')
init_bbox = (IMAGE_WIDTH//2 - (BBOX_WIDTH//2), IMAGE_HEIGHT//2 - (BBOX_HEIGHT//2), BBOX_HEIGHT, BBOX_WIDTH)
tracker_init = False



# ----- SERIAL SETUP -----

if ENABLE_LCD:
    lcd_monitor = serial.Serial(MONITOR_PORT, BAUD)
if ENABLE_MOTOR_SIG:
    motor_port = serial.Serial(ADAM_PORT, BAUD)



# ----- OUTPUT SETUP -----

if WRITE_OUTPUT:
    with open(output_dir, 'w') as o:
        o.write(f"OUTPUT for LIVE: {LIVE_FEED}   INPUT: {input_dir}\n")



# ----- MAIN LOOP -----

try:
    tic = time.time()
    while True:

        # Get RealSense Images
        frames = pipeline.wait_for_frames()
        error, col_img, dep_img = utils.format_frames(align, frames, depth_scale)
        if error:
            continue

        # Get YOLO Bounding Boxes 
        boxes, confs, clss = trt_yolo.detect(col_img, CONF_THRESH)
        boxes, confs, clss = boxes[clss==PERSON_CLASS], confs[clss==PERSON_CLASS], clss[clss==PERSON_CLASS]

        # Mask Colour Images
        person_mask = utils.person_masking(boxes, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH)
        depth_mask = utils.depth_masking(dep_img, clip_dist=CLIP_DIST)
        per_img = col_img*person_mask*depth_mask

        # Switch Tracker On/Off From Center Distance
        center_dist = utils.get_center_distance(dep_img)
        if (center_dist > DIST_THRESH) and (not tracker_init):
            tracker_init = True 
            ret = tracker.init(per_img, init_bbox)
        elif (center_dist < DIST_THRESH) and (tracker_init) and (center_dist > 1e-6):
            tracker_init = False

        # Update Tracker
        if tracker_init:
            ret, bbox = tracker.update(per_img)

        # Calculate Signals of Interest
        if tracker_init and ret and bbox:
            bbox_roi = bbox*(np.array(bbox) > 0)
            d = np.percentile(dep_img[int(bbox_roi[1]):int(bbox_roi[1]+bbox_roi[3]), int(bbox_roi[0]):int(bbox_roi[0]+bbox_roi[2])], BBOX_QMIN)
            a = ((bbox_roi[0] + bbox_roi[2]//2) - IMAGE_WIDTH//2) * IMAGE_LFOV_DEG / IMAGE_WIDTH
        else:
            d = None
            a = None

        # Write LCD Feedback
        if ENABLE_LCD:
            if tracker_init:
                lcd_monitor.write(f"{np.round(d,2)} m,{np.round(a,0)} deg\n".encode('utf-8'))
            else: 
                lcd_monitor.write(f"Patient Seated\n".encode('utf-8'))

        # Write Motor Signals
        if ENABLE_MOTOR_SIG:
            motor_port.write(f"{UI_STATE},{int(not tracker_init)},{np.round(d,4)},{np.round(a,4)}\n".encode('utf-8'))

        # Display Window
        if DISP:
            if tracker_init:
                img = vis.draw_bboxes(per_img, boxes, confs, clss)
                if ret and bbox:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
                img = show_fps(img, fps)
                cv2.imshow('RealSense Sensors', img)
            else:
                img = show_fps(col_img, fps)
                cv2.imshow('RealSense Sensors', img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

        # Write Log File
        if WRITE_OUTPUT:
            if tracker_init:
                if bbox:
                    with open(output_dir, "a") as f:
                       print(f"{frames.get_timestamp()},{UI_STATE},{int(not tracker_init)},{d},{a},{fps},{bbox}", file=f)
            else:
                with open(output_dir, "a") as f:
                    print(f"{frames.get_timestamp()},{UI_STATE},{int(not tracker_init)},{d},{a},{fps},()", file=f)
        
        # Update FPS
        fps, tic = utils.update_fps(fps, tic)

finally:
    cv2.destroyAllWindows()
    pipeline.stop()
    if ENABLE_LCD:
        lcd_monitor.close()
    if ENABLE_MOTOR_SIG:
        motor_port.close()

