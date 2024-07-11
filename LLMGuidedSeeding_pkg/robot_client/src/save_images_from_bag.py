#!/usr/bin/env python
import os
import rosbag
from cv_bridge import CvBridge
import cv2
import argparse

parser = argparse.ArgumentParser(description='Extract images from a ROS bag file.')
parser.add_argument('--bag_path', type=str, required=True, help='The path to your ROS bag file')
parser.add_argument('--output_dir', type=str, required=True, help='The directory to save the images')

args = parser.parse_args()

# The path to your ROS bag file
bag_path = args.bag_path
output_dir = args.output_dir

# Topics to extract
topics = [
    '/H03/front_cam/image_color/compressed',
    '/H03/left_cam/image_color/compressed',
    '/H03/right_cam/image_color/compressed'
]

# Folders to save images
folders = {
    '/H03/front_cam/image_color/compressed': 'front',
    '/H03/left_cam/image_color/compressed': 'left',
    '/H03/right_cam/image_color/compressed': 'right'
}

# Ensure output directories exist
for folder in folders.values():
    path = os.path.join(output_dir, folder)
    if not os.path.exists(path):
        os.makedirs(path)

# Initialize CV bridge
bridge = CvBridge()

# Process the bag file
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=topics):
        # Convert ROS Image message to OpenCV image
        try:
            cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            print(f"Error converting image: {e}")
            continue

        # Generate image filename based on the timestamp and topic
        timestamp = t.to_nsec()
        folder = folders[topic]
        path = os.path.join(output_dir, folder)
        filename = os.path.join(path, f"{timestamp}.jpg")

        # Save the image
        cv2.imwrite(filename, cv_image)
        print(f"Saved {filename}")

print("Extraction complete.")
