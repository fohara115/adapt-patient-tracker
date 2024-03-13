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
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=200)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
#matcher = cv2.BFMatcher()
init_patient_bbox = (IMAGE_WIDTH//2 - (BBOX_WIDTH//2), IMAGE_HEIGHT//2 - (BBOX_HEIGHT//2), IMAGE_WIDTH//2 + (BBOX_WIDTH//2), IMAGE_HEIGHT//2 + (BBOX_HEIGHT//2))




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
            while True:
                kp_p, des_p = surf.detectAndCompute(utils.cut_bbox(col_img,init_patient_bbox), None)
                if len(kp_p)>0:
                    break
            #print(des_p)
            #print(des_p.shape)
            #print(type(des_p))
            patient_bbox = init_patient_bbox
            '''print(len(kp_p))
            print(des_p)
            print(des_p[0])
            img = vis.draw_bboxes(utils.cut_bbox(col_img,init_patient_bbox), boxes, confs, clss)
            img = show_fps(img, fps)
            img = cv2.drawKeypoints(img, kp_p, None, (255,0,0), 4)
            cv2.imshow('RealSense Sensors', img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
            time.sleep(20)
            break'''
          
        elif (center_dist < DIST_THRESH) and (tracker_init) and (center_dist > 1e-6):
            tracker_init = False


      
        # Update Tracker if Person is Detected
        if tracker_init and (len(clss) > 0) and len(kp_p) > 0 and len(boxes)>0:
            best = 0
            best_i = None
            best_kp = None
            best_des = None
            for i, b in enumerate(boxes):
                roi = utils.cut_bbox(col_img,b)
                kp, des = surf.detectAndCompute(roi, None)
                if abs(len(kp) - len(kp_p)) < 120:
                    if (des_p is None) or (des is None):
                        continue

                    if (len(des_p) < 3) or (len(des) < 3):
                        continue
                    #print(f"{des_p.shape}   {des.shape}")
                    
                    nn_matches = matcher.knnMatch(des_p, des, 2)
                
                    num_match = 0
                    for m,n in nn_matches:
                        if m.distance < 0.7*n.distance:
                            num_match = num_match + 1
                    score = num_match/len(kp_p)

                    if (score >= best):
                        best = score
                        best_i = i
                        best_kp = kp
                        best_des = des

            #update 
            if best_kp is not None:
                #print(f"kpp{len(kp_p)}, kp{len(best_kp)}, {best}")
                kp_p = best_kp 
                des_p = best_des
                patient_bbox = boxes[best_i]

          'IDEAS: bbox area feature, custom knn, '
            


                  





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
            if tracker_init and (len(clss) > 0) and len(kp_p) > 0:
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


