import cv2
import Jetson.GPIO as GPIO
import numpy as np
import os
import pycuda.autoinit
import pyrealsense2 as rs
import serial
import time
import utils
import yaml

from collections import deque
from scipy.spatial import distance
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
ENABLE_UI_INPUT = cfg['project']['enable_ui_input']
ENABLE_D_SIG = cfg['project']['enable_d_signal']
ENABLE_A_SIG = cfg['project']['enable_a_signal']
CLIP_DIST = cfg['video']['clip_limit']
IMAGE_HEIGHT = cfg['video']['img_height']
IMAGE_WIDTH = cfg['video']['img_width']
IMAGE_CHANNELS = cfg['video']['img_channels']
IMAGE_LFOV_DEG = cfg['video']['img_wide_fov_deg']
DIST_THRESH = cfg['parameters']['seated_trigger']
DEF_UI_STATE = cfg['parameters']['init_ui_state']

MAX_ARR = cfg['tracker']['max_arr']
BBOX_WIDTH = cfg['tracker']['bbox_width']
BBOX_HEIGHT = cfg['tracker']['bbox_height']
POP_PER = cfg['tracker']['pop_period']
QUEUE_LEN = cfg['tracker']['queue_length']
NFEAT = cfg['tracker']['orb_feats']
BBOX_MIN = cfg['tracker']['bbox_min']
BBOX_QMIN = cfg['tracker']['bbox_qmin']
BBOX_MIN = cfg['tracker']['bbox_min']

MODEL = cfg['model']['name']
CAT_NUM = cfg['model']['cat_num']
PERSON_CLASS = cfg['model']['person_label']
LETTER_BOX = cfg['model']['letter_box']
CONF_THRESH = cfg['model']['conf_thr']
B0_PIN = cfg['gpio']['byte0']
B1_PIN = cfg['gpio']['byte1']
LED_PIN = cfg['gpio']['safety_light']
BAUD = cfg['serial']['baud_rate']
MONITOR_PORT = cfg['serial']['lcd_port']
D_PORT = cfg['serial']['d_port']
A_PORT = cfg['serial']['a_port']
WRITE_OUTPUT = cfg['output']['log_output']
STATE_FILE = cfg['output']['state_file']
OUTPUT_ROOT = cfg['output']['output_root']



# ----- CLI SETUP -----





# ----- SERIAL SETUP -----

if ENABLE_LCD:
    lcd_monitor = serial.Serial(MONITOR_PORT, BAUD)
    utils.lcd_boot_msg(lcd_monitor)
if ENABLE_D_SIG:
    d_port = serial.Serial(D_PORT, BAUD)
if ENABLE_A_SIG:
    a_port = serial.Serial(A_PORT, BAUD)



# ----- DETECTOR SETUP -----

cls_dict = get_cls_dict(CAT_NUM)
vis = BBoxVisualization(cls_dict)
trt_yolo = TrtYOLO(MODEL, CAT_NUM, LETTER_BOX)
prev_mask1 = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), np.uint8)
prev_mask2 = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), np.uint8)



# ----- TRACKER SETUP -----

X = deque()
max_x =  np.array(MAX_ARR)
orb = cv2.ORB_create(nfeatures=NFEAT)
tracker_init = False
missing = False
poptime = 0
init_patient_bbox = (IMAGE_WIDTH//2 - (BBOX_WIDTH//2), IMAGE_HEIGHT//2 - (BBOX_HEIGHT//2), IMAGE_WIDTH//2 + (BBOX_WIDTH//2), IMAGE_HEIGHT//2 + (BBOX_HEIGHT//2))
d = None
a = None



# ----- GPIO SETUP -----

if ENABLE_UI_INPUT:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(B0_PIN, GPIO.IN)
    GPIO.setup(B1_PIN, GPIO.IN)



# ----- VIDEO SETUP -----

pipeline, config = utils.load_live_stream() if LIVE_FEED else utils.load_bag_file(input_dir)
profile = pipeline.start(config)
depth_scale = utils.get_depth_scale(profile)
align = rs.align(rs.stream.color)
fps = 0
t_prev = 0


# ----- UI RESET SETUP -----

prev_ui_state = 0
ui_state = 0


# ----- MAIN LOOP -----

try:
    tic = time.time()
    while True:

        # Update UI State
        if ENABLE_UI_INPUT:
            ui_state = utils.read_gpio_state(B0_PIN, B1_PIN)
        else: 
            ui_state = DEF_UI_STATE

        # Search for OFF->ON Change
        if (prev_ui_state==0) and (ui_state>0):
            # Create filename
            with open(STATE_FILE, 'r') as f:
                content = f.read()
                try:
                    fn_state = int(content.strip())
                except:
                    fn_state = "???"

            input_dir, output_dir = utils.process_cli_args_wstate(iroot=INPUT_ROOT, oroot=OUTPUT_ROOT, default=DEFAULT_VID, live=LIVE_FEED, state=fn_state)


            # Create the file
            if WRITE_OUTPUT:
                with open(output_dir, 'w') as o:
                o.write(f"OUTPUT for LIVE: {LIVE_FEED}   INPUT: {input_dir}\n")

        # Search for OFF->ON Change
        if (prev_ui_state>0) and (ui_state==0):
            # Update state file
            with open(STATE_FILE, 'r') as f:
                content = f.read()
                try:
                    fn_state = int(content.strip())
                except:
                    fn_state = "???"

            with open(STATE_FILE, 'w') as f:
                try:
                    fn_state = fn_state + 1
                except:
                    fn_state = "???"

                f.write('%d' % fn_state)
            


        # Update LCD
        if ENABLE_LCD:
            utils.update_lcd_board_state(lcd_monitor, ui_state)
            estop = utils.check_estop(lcd_monitor)
            if estop:
                if ENABLE_D_SIG:
                    utils.send_d_stop(d_port)
                if ENABLE_A_SIG:
                    utils.send_a_stop(a_port)
                break
            utils.update_lcd_display(lcd_monitor, tracker_init, d, a, missing, ui_state, fps)
            

        # Get RealSense Images
        frames = pipeline.wait_for_frames()
        t = frames.get_timestamp()
        if t_prev > t:
            break
        t_prev = t
        error, col_img, dep_img = utils.format_frames(align, frames, depth_scale)
        if error:
            continue


        # Get YOLO Bounding Boxes 
        boxes, confs, clss = trt_yolo.detect(col_img, CONF_THRESH)
        boxes, confs, clss = boxes[clss==PERSON_CLASS], confs[clss==PERSON_CLASS], clss[clss==PERSON_CLASS]


        # Mask Colour Images
        depth_mask = utils.depth_masking(dep_img, clip_dist=CLIP_DIST)
        fimg = col_img*depth_mask


        # Switch Tracker On/Off From Center Distance
        center_dist = utils.get_center_distance(dep_img)
        if (center_dist > DIST_THRESH) and (not tracker_init) and (len(clss) > 0):
            tracker_init = True 
            x1 = utils.get_features_v4(orb, fimg, init_patient_bbox)
            X.appendleft(x1) 
            patient_bbox = init_patient_bbox
          
        elif (center_dist < DIST_THRESH) and (tracker_init) and (center_dist > 1e-6):
            tracker_init = False
            X = deque()
      

        # Update Tracker if Person is Detected
        if tracker_init and (len(clss) > 0) and len(boxes)>0:
            best_d = 10000
            best_i = None
            best_x = None
            for i, b in enumerate(boxes):
                x = utils.get_features_v4(orb, fimg, b)
                if len(X) < QUEUE_LEN:
                    X.appendleft(x)
                sx = x / max_x
                total_d = 0
                for xv in X:
                    dist = distance.cityblock(xv / max_x, sx)
                    total_d = total_d + dist

                if total_d <= best_d:
                    best_d = total_d
                    best_i = i
                    best_x = x
            patient_bbox = boxes[best_i]

            if (t - poptime > POP_PER):
                X.pop()
                X.appendleft(best_x)
                poptime = t


        # Calculate Signals of Interest
        if tracker_init and (patient_bbox is not None) and (len(clss) > 0):
            p1 = (patient_bbox[0], patient_bbox[1])
            p2 = (patient_bbox[0]+patient_bbox[2], patient_bbox[1]+patient_bbox[3])
            d = utils.calculate_dist_from_roi(dep_img, p1, p2, BBOX_MIN, BBOX_QMIN)
            a = utils.calculate_ang(p1, p2, IMAGE_WIDTH, IMAGE_LFOV_DEG)
        else:
            d = None
            a = None


        # Write Motor Signals
        if ENABLE_D_SIG:
            utils.send_d_signals(d_port, d, ui_state, tracker_init, missing)
            utils.send_a_signals(d_port, a, d, ui_state, tracker_init, missing)
        if ENABLE_A_SIG:
            utils.send_a_signals(a_port, a, d, ui_state, tracker_init, missing)
            utils.send_d_signals(a_port, d, ui_state, tracker_init, missing)


        # Display Window
        if DISP:
            img = show_fps(col_img, fps)
            img = vis.draw_bboxes(img, boxes, confs, clss)
            if tracker_init and (len(clss) > 0):
                img = cv2.circle(img, ((patient_bbox[0]+patient_bbox[2])//2, IMAGE_HEIGHT//2), 25, (0,0,255),5)
                img = cv2.rectangle(img, (patient_bbox[0], patient_bbox[1]), (patient_bbox[2], patient_bbox[3]), (0, 0, 255), 5)
            cv2.imshow('RealSense Sensors', col_img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break


        # Write Log File
        if WRITE_OUTPUT:
            if tracker_init:
                if patient_bbox is not None:
                    with open(output_dir, "a") as f:
                       ss = f"{t}|{ui_state}|{int(not tracker_init)}|{d}|{a}|{fps}|{patient_bbox[0]}|{patient_bbox[1]}|{patient_bbox[2]}|{patient_bbox[3]}|{len(confs)}|x"
                       
                       s1 = ""
                       for i, b in enumerate(boxes):
                           x = utils.get_features_v4(orb, fimg, b)
                           x = x / max_x

                           if len(x) != 9:
                               s1 = s1 + "|||||||||"
                           else:
                               s1 = s1 + f"|{x[0]}|{x[1]}|{x[2]}|{x[3]}|{x[4]}|{x[5]}|{x[6]}|{x[7]}|{x[8]}"
                       for _ in range(10 - len(boxes)):
                           s1 = s1 + "|||||||||"

                       s2 = '|Q'

                       s3 = ""
                       for i in range(20):
                           if (len(X)>(i)):
                               if len(X[i]) != 9:
                                   s3 = s3 + "|||||||||"
                               else:
                                   x = X[i] / max_x
                                   s3 = s3 + f"|{x[0]}|{x[1]}|{x[2]}|{x[3]}|{x[4]}|{x[5]}|{x[6]}|{x[7]}|{x[8]}"
                           else:
                               s3 = s3 + "|||||||||"

                           
                       print(ss+s1+s2+s3, file=f)
            else:
                with open(output_dir, "a") as f:
                    print(f"{t}|{ui_state}|{int(not tracker_init)}|{d}|{a}|{fps}|None|None|None|None|{len(confs)}|None|None", file=f)
        
        
        # Update FPS
        fps, tic = utils.update_fps(fps, tic)
        prev_ui_state = ui_state

finally:
    

    cv2.destroyAllWindows()
    pipeline.stop()
    if ENABLE_LCD:
        utils.lcd_shutdown_msg(lcd_monitor)
        lcd_monitor.close()
    if ENABLE_D_SIG:
        d_port.close()
    if ENABLE_A_SIG:
        a_port.close()

    while len(X)>0:
        X.pop()
    del X

