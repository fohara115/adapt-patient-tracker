import pyrealsense2 as rs
import numpy as np
import cv2
#import yaml
import getopt
import sys
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

#cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.CLoader)
LIVE_FEED = True
DISP = True
CAT_NUM = 80
PERSON_CLASS = 0
MODEL = 'yolov4-tiny'
LETTER_BOX = True
CONF_THRESH = 0.3
ENABLE_LCD = True
BAUD = 9600
MONITOR_PORT = '/dev/ttyUSB0'
ROOT = '/media/fohara/645E941F5E93E856/bme/data/'
DEFAULT_VID = '20231124_163447.bag'

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_LFOV_DEG = 42
BBOX_HEIGHT = 150
BBOX_WIDTH = 200
DIST_THRESH = 0.75
BBOX_QMIN = 15

##from args
argv = sys.argv[1:]
opts, args = getopt.getopt(argv, 'e')
if (len(args)>0):
    filename = ROOT+args[0]
else:
    filename = ROOT+DEFAULT_VID


# ----- VIDEO SETUP -----

pipeline, config = utils.load_live_stream() if LIVE_FEED else utils.load_bag_file(filename)
profile = pipeline.start(config)
depth_scale = utils.get_depth_scale(profile)
align = rs.align(rs.stream.color)


# ----- DETECTOR SETUP -----

cls_dict = get_cls_dict(CAT_NUM)
vis = BBoxVisualization(cls_dict)
trt_yolo = TrtYOLO(MODEL, CAT_NUM, LETTER_BOX)


# ----- TRACKER SETUP -----

tracker = cv2.TrackerCSRT_create() #cv2.TrackerKCF_create(), cv2.TrackerMIL_create()
init_bbox = (IMAGE_WIDTH//2 - (BBOX_WIDTH//2), IMAGE_HEIGHT//2 - (BBOX_HEIGHT//2), BBOX_HEIGHT, BBOX_WIDTH)
tracker_init = False




# ----- SERIAL SETUP -----

if ENABLE_LCD:
    lcd_monitor = serial.Serial(MONITOR_PORT, BAUD) ##ser.write(b"Testing Testing\n")


# ----- MAIN LOOP -----
try:
    fps = 0.0
    tic = time.time()
    while True:
        frames = pipeline.wait_for_frames()
        t = frames.get_timestamp()

    
        # Align images
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            print('Skipping problematic frame...')
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if depth_image.size == 0:
            continue
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        


        # YOLO
        boxes, confs, clss = trt_yolo.detect(color_image, CONF_THRESH)
        boxes, confs, clss = boxes[clss==PERSON_CLASS], confs[clss==PERSON_CLASS], clss[clss==PERSON_CLASS]
        img = vis.draw_bboxes(color_image, boxes, confs, clss)
        img = show_fps(img, fps)


        # MASKING
        mask = np.zeros(color_image.shape, np.uint8)
        for xmin, ymin, xmax, ymax in boxes:
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (1, 1, 1), -1)
        fcimg = color_image*mask

        # MIL TRACKER
        center_dist = depth_image[IMAGE_HEIGHT//2, IMAGE_WIDTH//2] * depth_scale
        if (center_dist > DIST_THRESH) and (not tracker_init): # Switch tracker on
            ret = tracker.init(color_image, init_bbox)
            tracker_init = True 
        elif (center_dist < DIST_THRESH) and (tracker_init) and (center_dist > 1e-6): # Switch tracker off
            tracker_init = False
            print('too close! turning off')

        if tracker_init:
            ret, bbox = tracker.update(fcimg)
            if ret and DISP:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(fcimg, p1, p2, (255,0,0), 2, 1)

            if ret:
                out_dist = np.percentile(depth_image[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] * depth_scale, BBOX_QMIN)
                out_angle = ((bbox[0] + bbox[2]//2) - IMAGE_WIDTH//2) * IMAGE_LFOV_DEG / IMAGE_WIDTH
                #print(f"{out_dist}, {out_angle}")


        # DISP
        #images = np.hstack((img, depth_colormap))
        fcimg = show_fps(fcimg, fps)
        if DISP:
            if tracker_init:
                cv2.imshow('RealSense Sensors', fcimg)
            else:
                cv2.imshow('RealSense Sensors', color_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

        # LCD
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




