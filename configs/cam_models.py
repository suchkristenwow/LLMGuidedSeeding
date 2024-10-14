from sensor_msgs.msg import CameraInfo
from image_geometry import PinholeCameraModel

# LEFT #############################################################################################
# Step 1: Create and populate the CameraInfo message
camera_info = CameraInfo()
camera_info.header.seq = 439
camera_info.header.stamp.secs = 1724339914
camera_info.header.stamp.nsecs = 690719588
camera_info.header.frame_id = "cam_left_link"

camera_info.height = 608
camera_info.width = 808
camera_info.distortion_model = "plumb_bob"
camera_info.D = [-0.243231, 0.053397, -0.001172, 0.000829, 0.0]
camera_info.K = [490.48909, 0.0, 413.2397, 0.0, 492.63605, 325.99985, 0.0, 0.0, 1.0]
camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
camera_info.P = [393.83435, 0.0, 419.88234, 0.0, 0.0, 440.93787, 330.55257, 0.0, 0.0, 0.0, 1.0, 0.0]

camera_info.binning_x = 0
camera_info.binning_y = 0

camera_info.roi.x_offset = 0
camera_info.roi.y_offset = 0
camera_info.roi.height = 0
camera_info.roi.width = 0
camera_info.roi.do_rectify = False

# Step 2: Initialize the PinholeCameraModel
left_camera_model = PinholeCameraModel()
left_camera_model.fromCameraInfo(camera_info) 

# RIGHT #############################################################################################
camera_info = CameraInfo()
camera_info.header.seq = 439
camera_info.header.stamp.secs = 1724339914
camera_info.header.stamp.nsecs = 690719588
camera_info.header.frame_id = "cam_right_link"

camera_info.height = 608
camera_info.width = 808
camera_info.distortion_model = "plumb_bob"
camera_info.D = [-0.243231, 0.053397, -0.001172, 0.000829, 0.0]
camera_info.K = [496.36019, 0.0, 389.41303, 0.0, 495.20248, 301.6506, 0.0, 0.0, 1.0]
camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
camera_info.P = [402.74881, 0.0, 385.66006, 0.0, 0.0, 440.9407, 301.50641, 0.0, 0.0, 0.0, 1.0, 0.0]

camera_info.binning_x = 0
camera_info.binning_y = 0

camera_info.roi.x_offset = 0
camera_info.roi.y_offset = 0
camera_info.roi.height = 0
camera_info.roi.width = 0
camera_info.roi.do_rectify = False

# Step 2: Initialize the PinholeCameraModel
right_camera_model = PinholeCameraModel()
right_camera_model.fromCameraInfo(camera_info) 

# FRONT #############################################################################################
camera_info = CameraInfo()
camera_info.header.seq = 439
camera_info.header.stamp.secs = 1724339914
camera_info.header.stamp.nsecs = 690719588
camera_info.header.frame_id = "cam_front_link"

camera_info.height = 608
camera_info.width = 808
camera_info.distortion_model = "plumb_bob"
camera_info.D = [-0.25861, 0.06925, 0.001231, 0.001145, 0.0]
camera_info.K = [510.1259, 0.0, 420.6644, 0.0, 509.4401, 306.44069, 0.0, 0.0, 1.0]
camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
camera_info.P = [417.10114, 0.0, 429.82936, 0.0, 0.0, 458.0275, 308.05336, 0.0, 0.0, 0.0, 1.0, 0.0] 

camera_info.binning_x = 0
camera_info.binning_y = 0

camera_info.roi.x_offset = 0
camera_info.roi.y_offset = 0
camera_info.roi.height = 0
camera_info.roi.width = 0
camera_info.roi.do_rectify = False

# Step 2: Initialize the PinholeCameraModel
front_camera_model = PinholeCameraModel()
front_camera_model.fromCameraInfo(camera_info) 