import pyrealsense2 as rs
import numpy as np
import os.path
import cv2

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
    config.enable_stream(rs.stream.depth, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color)

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
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)

    return pipeline, config


