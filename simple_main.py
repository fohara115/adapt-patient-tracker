import pyrealsense2 as rs
import numpy as np
import cv2
#import yaml
import getopt
import sys
#import serial
import utils
import os
import pycuda.autoinit 

from trt_utils.yolo_classes import get_cls_dict
from trt_utils.camera import add_camera_args, Camera
from trt_utils.display import open_window, set_display, show_fps
from trt_utils.visualization import BBoxVisualization
from trt_utils.yolo_with_plugins import TrtYOLO



# ----- LOAD CONFIG & ARGS -----

#cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.CLoader)
# Argument parsing

CAT_NUM = 80
MODEL = 'yolov4-tiny'
LETTER_BOX = True
CONF_THRESH = 0.3
BAUD = 9600
MONITOR_PORT = '/dev/ttyUSB0'


# ----- VIDEO SETUP -----

pipeline, config = utils.load_live_stream() #pipeline, config = load_bag_file(filename)
profile = pipeline.start(config)
depth_scale = utils.get_depth_scale(profile)
align = rs.align(rs.stream.color)


# ----- DETECTOR SETUP -----

cls_dict = get_cls_dict(CAT_NUM)
#trt_yolo = TrtYOLO(MODEL, CAT_NUM, LETTER_BOX)


# ----- TRACKER SETUP -----

#tracker = cv2.TrackerMIL_create() #BROKEN, needs install


# ----- SERIAL SETUP -----

#lcd_monitor = serial.Serial(MONITOR_PORT, BAUD) ##ser.write(b"Testing Testing\n")








