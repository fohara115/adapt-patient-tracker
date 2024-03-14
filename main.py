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
#from sklearn.preprocessing import StandardScaler
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
TRACKER_TYPE = cfg['tracker']['type']
BBOX_HEIGHT = cfg['tracker']['bbox_height']
BBOX_WIDTH = cfg['tracker']['bbox_width']
ROI_WIDTH = cfg['tracker']['roi_width']
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
OUTPUT_ROOT = cfg['output']['output_root']

input_dir, output_dir = utils.process_cli_args(iroot=INPUT_ROOT, oroot=OUTPUT_ROOT, default=DEFAULT_VID, live=LIVE_FEED)

if not LIVE_FEED:
    print(f"{input_dir}...")



# ----- DETECTOR SETUP -----

cls_dict = get_cls_dict(CAT_NUM)
vis = BBoxVisualization(cls_dict)
trt_yolo = TrtYOLO(MODEL, CAT_NUM, LETTER_BOX)
prev_mask1 = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), np.uint8)
prev_mask2 = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), np.uint8)



# ----- TRACKER SETUP -----

if TRACKER_TYPE=='CSRT':
    tracker = cv2.TrackerCSRT_create()
elif TRACKER_TYPE=='KCF':
    tracker = cv2.TrackerKCF_create()
elif TRACKER_TYPE=='MIL':
    tracker = cv2.TrackerMIL_create()
else:
    raise(f'ERROR: Provided tracker type {TRACKER_TYPE} is not supported in this project.')
init_bbox = (IMAGE_WIDTH//2 - (BBOX_WIDTH//2), IMAGE_HEIGHT//2 - (BBOX_HEIGHT//2), BBOX_WIDTH, BBOX_HEIGHT)
tracker_init = False
missing = False



# ----- SERIAL SETUP -----

if ENABLE_LCD:
    lcd_monitor = serial.Serial(MONITOR_PORT, BAUD)
if ENABLE_D_SIG:
    d_port = serial.Serial(D_PORT, BAUD)
if ENABLE_A_SIG:
    a_port = serial.Serial(A_PORT, BAUD)



# ----- GPIO SETUP -----

if ENABLE_UI_INPUT:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(B0_PIN, GPIO.IN)
    GPIO.setup(B1_PIN, GPIO.IN)




# ----- OUTPUT SETUP -----

if WRITE_OUTPUT:
    with open(output_dir, 'w') as o:
        o.write(f"OUTPUT for LIVE: {LIVE_FEED}   INPUT: {input_dir}\n")



# ----- VIDEO SETUP -----

pipeline, config = utils.load_live_stream() if LIVE_FEED else utils.load_bag_file(input_dir)
profile = pipeline.start(config)
depth_scale = utils.get_depth_scale(profile)
align = rs.align(rs.stream.color)
fps = 0
t_prev = 0



# ----- SURF SETUP -----

hess_thresh = 1000
surf = cv2.xfeatures2d.SURF_create(hess_thresh)
surf.setUpright(False)
#index_params = dict(algorithm=1, trees=5)
#search_params = dict(checks=200)
#matcher = cv2.FlannBasedMatcher(index_params, search_params)
#matcher = cv2.BFMatcher()
init_patient_bbox = (IMAGE_WIDTH//2 - (BBOX_WIDTH//2), IMAGE_HEIGHT//2 - (BBOX_HEIGHT//2), IMAGE_WIDTH//2 + (BBOX_WIDTH//2), IMAGE_HEIGHT//2 + (BBOX_HEIGHT//2))

X = deque()
poptime = 0
pop_period = 500 #ms
#max_x =  np.array([1,1,1,1,1,1,1,1,1])#np.array([360, 7e5, 255, 255, 255])
#max_x =  np.array([500,500**2,600,700,600*700,600,200,200,200])
max_x =  np.array([480,640,480,600,600,255,255,255,1])
queue_len = 5
#scaler = StandardScaler()

orb = cv2.ORB_create(nfeatures=500)
#index_params = dict(algorithm=1, trees=5)
#search_params = dict(checks=200)
#matcher = cv2.FlannBasedMatcher(index_params, search_params)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)




# ----- MAIN LOOP -----

try:
    tic = time.time()
    while True:

        # Get UI Input
        if ENABLE_UI_INPUT:
            ui_state = utils.read_gpio_state(B0_PIN, B1_PIN)
        else: 
            ui_state = DEF_UI_STATE
        if ENABLE_LCD:
            utils.update_lcd_board_state(lcd_monitor, ui_state)

        # Check ESTOP
        if ENABLE_LCD:
            estop = utils.check_estop(lcd_monitor)
            if estop:
                if ENABLE_D_SIG:
                    utils.send_d_stop(d_port)
                if ENABLE_A_SIG:
                    utils.send_a_stop(a_port)
                break
            

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
        if (center_dist > DIST_THRESH) and (not tracker_init):
            tracker_init = True 
            #roi = utils.cut_bbox(img, init_patient_bbox)
            #kp_p, des_p = orb.detectAndCompute(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), None)
            x1 = utils.get_features_v4(orb, fimg, init_patient_bbox)
            for _ in range(queue_len):
                X.appendleft(x1)

            #centroid = np.mean(np.array(X) / max_x, axis=0)
            patient_bbox = init_patient_bbox
          
        elif (center_dist < DIST_THRESH) and (tracker_init) and (center_dist > 1e-6):
            tracker_init = False


      
        # Update Tracker if Person is Detected
        if tracker_init and (len(clss) > 0) and len(boxes)>0:
            best_d = 10000
            best_i = None
            best_x = None
            for i, b in enumerate(boxes):
                x = utils.get_features_v4(orb, fimg, b)
                sx = x / max_x
                total_d = 0
                for xv in X:
                    dist = distance.cityblock(xv / max_x, sx)
                    total_d = total_d + dist
                print(f"{np.round(total_d, 4)}:  {sx}")

                '''# matching score
                if len(des) > 2 and len(des_p) > 2:
                    matches = matcher.match(des, des_p)
                    good_matches = [m for m in matches if m.distance < 24]
                    mscore = len(good_matches) / min(len(des), len(des_p))
                    print(f"{np.round(total_d, 4)}: {mscore}")
                    '
                    #total_d = total_d - 0.05*mscore'''

                if total_d <= best_d:
                    best_d = total_d
                    best_i = i
                    best_x = x
                    #des_p = des
            
            print(f"best_d: {best_d}")
            patient_bbox = boxes[best_i]

            if (t - poptime > pop_period):
                print('update')
                X.pop()
                X.appendleft(best_x)
                #centroid = np.mean(np.array(X) / max_x, axis=0)
               # max_x = np.max(np.array(X), axis=0)
                print(max_x)
                poptime = t



        '''if tracker_init:
            num_people = len(clss)
            if num_people == 0:
                missing = True
                ret = False
            elif (num_people > 0) and missing:
                #if np.sum(bbox) != 0:
                missing = False
                ret, bbox = tracker.update(per_img)
            else:
                ret, bbox = tracker.update(per_img)'''

        # Calculate Signals of Interest
        '''if tracker_init and ret and bbox and not missing:
            p1, p2 = utils.full_height_box(bbox, IMAGE_HEIGHT, IMAGE_WIDTH, width=ROI_WIDTH)
            d = utils.calculate_dist_from_roi(dep_img, p1, p2, BBOX_MIN, BBOX_QMIN)
            a = utils.calculate_ang(p1, p2, IMAGE_WIDTH, IMAGE_LFOV_DEG)
        else:'''
        d = None
        a = None

        # Write LCD Feedback
        if ENABLE_LCD:
            utils.update_lcd_display(lcd_monitor, tracker_init, d, a, missing, ui_state, fps)
            
        # Write Motor Signals
        if ENABLE_D_SIG:
            utils.send_d_signals(d_port, d, ui_state, tracker_init, missing)

        if ENABLE_A_SIG:
            utils.send_a_signals(a_port, a, ui_state, tracker_init, missing)

        # Display Window
        if DISP:
            img = vis.draw_bboxes(col_img, boxes, confs, clss)
            img = show_fps(img, fps)
            if tracker_init and (len(clss) > 0):
                #img = cv2.drawKeypoints(img, kp_p, None, (255,0,0), 4)
                img = cv2.rectangle(img, (patient_bbox[0], patient_bbox[1]), (patient_bbox[2], patient_bbox[3]), (0, 0, 255), 5)
            cv2.imshow('RealSense Sensors', img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

        # Write Log File
        if WRITE_OUTPUT:
            if tracker_init:
                if ret and bbox:
                    with open(output_dir, "a") as f:
                       print(f"{t},{ui_state},{int(not tracker_init)},{d},{a},{fps},{bbox}", file=f)
            else:
                with open(output_dir, "a") as f:
                    print(f"{t},{ui_state},{int(not tracker_init)},{d},{a},{fps},()", file=f)
        
        # Update FPS
        fps, tic = utils.update_fps(fps, tic)

finally:
    cv2.destroyAllWindows()
    pipeline.stop()
    if ENABLE_LCD:
        lcd_monitor.close()
    if ENABLE_D_SIG:
        d_port.close()
    if ENABLE_A_SIG:
        a_port.close()

    while len(X)>0:
        X.pop()
    del X


