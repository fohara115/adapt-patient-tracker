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

from skimage.feature import haar_like_feature
from skimage.transform import integral_image

from trt_utils.yolo_classes import get_cls_dict
from trt_utils.camera import add_camera_args, Camera
from trt_utils.display import open_window, set_display, show_fps
from trt_utils.visualization import BBoxVisualization
from trt_utils.yolo_with_plugins import TrtYOLO

cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.CLoader)
LIVE_FEED = cfg['input']['live_feed']
MODEL = cfg['model']['name']
CAT_NUM = cfg['model']['cat_num']
PERSON_CLASS = cfg['model']['person_label']
LETTER_BOX = cfg['model']['letter_box']
CONF_THRESH = cfg['model']['conf_thr']


cls_dict = get_cls_dict(CAT_NUM)
vis = BBoxVisualization(cls_dict)
trt_yolo = TrtYOLO(MODEL, CAT_NUM, LETTER_BOX)


pipeline, config = utils.load_live_stream() if LIVE_FEED else utils.load_bag_file(input_dir)
profile = pipeline.start(config)
depth_scale = utils.get_depth_scale(profile)
align = rs.align(rs.stream.color)
fps = 0


hess_thresh = 5000
surf = cv2.xfeatures2d.SURF_create(hess_thresh)
surf.setUpright(False)

#bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
#surf.setContrastThreshold(0.25)
#surf.setEdgeThreshold(5)
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

first = True

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

        if first:
            kp, des = surf.detectAndCompute(col_img, None)
            first = False

        
        kp_prev, des_prev = kp, des
        kp, des = surf.detectAndCompute(col_img, None)
        #kp, des = surf.compute(col_img, kp)
        #matches = bf.match(des_i, des)
        nn_matches = matcher.knnMatch(des_prev, des, 2)
        num_match = 0
        for i,(m,n) in enumerate(nn_matches):
            if m.distance < 0.7*n.distance:
                num_match = num_match + 1
        print(f"{num_match} / {len(kp)}")
        
        #print(len(kp))

        img = vis.draw_bboxes(col_img, boxes, confs, clss)
        img = show_fps(img, fps)
        img = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
        cv2.imshow('RealSense Sensors', img)
         

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
        
        # Update FPS
        fps, tic = utils.update_fps(fps, tic)

finally:
    cv2.destroyAllWindows()
    pipeline.stop()




