import pyrealsense2 as rs
import numpy as np
import cv2
import yaml
import serial
import utils
import os
import pycuda.autoinit 
import time

from trt_utils.yolo_classes import get_cls_dict
from trt_utils.camera import add_camera_args, Camera
from trt_utils.display import open_window, set_display, show_fps
from trt_utils.visualization import BBoxVisualization
from trt_utils.yolo_with_plugins import TrtYOLO

# sudo chmod a+rw /dev/ttyUSB0




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
OUTPUT_ROOT = cfg['output']['log_output']

input_dir, output_dir = utils.process_cli_args(iroot=INPUT_ROOT, oroot=OUTPUT_ROOT, default=DEFAULT_VID, live=LIVE_FEED)



# ----- VIDEO SETUP -----

pipeline, config = utils.load_live_stream() if LIVE_FEED else utils.load_bag_file(input_dir)
profile = pipeline.start(config)
depth_scale = utils.get_depth_scale(profile)
align = rs.align(rs.stream.color)



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
    


# ----- MAIN LOOP -----

try:
    fps = 0.0
    tic = time.time()
    while True:
        # Align images
        frames = pipeline.wait_for_frames()
        t = frames.get_timestamp()
        error, col_img, dep_img = utils.format_frames(align, frames, depth_scale)
        if error:
            continue

        # YOLO
        boxes, confs, clss = trt_yolo.detect(col_img, CONF_THRESH)
        boxes, confs, clss = boxes[clss==PERSON_CLASS], confs[clss==PERSON_CLASS], clss[clss==PERSON_CLASS]

        # MASKING
        person_mask = utils.person_masking(boxes, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH)
        depth_mask = utils.depth_masking(dep_img, clip_dist=CLIP_DIST)
        per_img = col_img*person_mask*depth_mask

        # MIL TRACKER
        center_dist = utils.get_center_distance(dep_img)
        if (center_dist > DIST_THRESH) and (not tracker_init): # Switch tracker on
            ret = tracker.init(per_img, init_bbox)
            tracker_init = True 
        elif (center_dist < DIST_THRESH) and (tracker_init) and (center_dist > 1e-6): # Switch tracker off
            tracker_init = False
            print('too close! turning off')

        if tracker_init:
            ret, bbox = tracker.update(per_img)
            if ret and DISP:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(per_img, p1, p2, (255,0,0), 2, 1)
            try:
                if ret:
                    out_dist = np.percentile(dep_img[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] * depth_scale, BBOX_QMIN)
                    out_angle = ((bbox[0] + bbox[2]//2) - IMAGE_WIDTH//2) * IMAGE_LFOV_DEG / IMAGE_WIDTH
                    #print(f"{out_dist}, {out_angle}")
            except:
                out_dist = None
                out_angle = None

        # DISP
        #images = np.hstack((img, depth_colormap))
        img = vis.draw_bboxes(per_img, boxes, confs, clss)
        img = show_fps(img, fps)
        if DISP:
            if tracker_init:
                cv2.imshow('RealSense Sensors', img)
            else:
                cv2.imshow('RealSense Sensors', col_img)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

        # LCD
        if ENABLE_LCD:
            if tracker_init and ret:
                lcd_monitor.write(f"{np.round(out_dist,2)} m,{np.round(out_angle,0)} deg\n".encode('utf-8'))
            else: 
                lcd_monitor.write(f"Patient Seated\n".encode('utf-8'))


        # FPS
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05) # calculate an exponentially decaying average of fps number
        tic = toc


finally:
    cv2.destroyAllWindows()
    pipeline.stop()
    if ENABLE_LCD:
        lcd_monitor.close()




