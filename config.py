import numpy as np

# --- Path Settings ---
VIDEO_INPUT_PATH = "sample_video.mp4"
VIDEO_OUTPUT_PATH = "output_video.avi"

# --- YOLO Model Settings ---
YOLO_MODEL_PATH = 'yolov8n.pt'  # Using a small, fast model
YOLO_CONFIDENCE_THRESHOLD = 0.4
# Classes to detect (COCO dataset class indices)
# 2: car, 3: motorcycle, 5: bus, 7: truck
YOLO_TARGET_CLASSES = [2, 3, 5, 7]

# --- Camera and Perspective Transformation Settings ---
# NOTE: These points are calibrated for the `sample_video.mp4` from archive.org.
# They define a trapezoidal region of interest on the road.
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

# Source points from the input image (trapezoid on the road)
# Format: [top-left, top-right, bottom-right, bottom-left]
# These points have been adjusted for the new video from archive.org.
SRC_POINTS = np.float32([
    [542, 383],  # Top-left
    [753, 382],  # Top-right
    [930, 550], # Bottom-right
    [404, 550]   # Bottom-left
])

# Destination points in the bird's-eye view (a rectangle)
# This defines the dimensions of the output bird's-eye view image.
DST_WIDTH = 480
DST_HEIGHT = 640
DST_POINTS = np.float32([
    [0, 0],              # Top-left
    [DST_WIDTH - 1, 0],  # Top-right
    [DST_WIDTH - 1, DST_HEIGHT - 1], # Bottom-right
    [0, DST_HEIGHT - 1]  # Bottom-left
])

# --- Path Planning and Collision Avoidance Settings ---
# The "ego" vehicle's properties in the bird's-eye view
EGO_VEHICLE_WIDTH = 2.5  # meters
EGO_VEHICLE_LENGTH = 4.5 # meters

# Convert meters to pixels for the bird's-eye view
# This is a crucial calibration step. Assume the DST_HEIGHT covers 50 meters of road.
METERS_PER_PIXEL_Y = 50 / DST_HEIGHT
METERS_PER_PIXEL_X = 3.7 / DST_WIDTH # Assuming a standard lane width of 3.7m

EGO_VEHICLE_WIDTH_PIXELS = EGO_VEHICLE_WIDTH / METERS_PER_PIXEL_X

# How far ahead the local path planner should look (in pixels in BEV)
PLANNING_HORIZON = int(DST_HEIGHT * 0.8)

# Safe distance to keep from other vehicles (in pixels in BEV)
SAFE_DISTANCE = 30 # pixels

# Lateral shift for avoidance maneuver (in pixels in BEV)
AVOIDANCE_SHIFT = int(EGO_VEHICLE_WIDTH_PIXELS * 1.2)
