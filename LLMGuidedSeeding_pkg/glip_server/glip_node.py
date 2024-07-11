#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from viper_navigation_client.msg import NavClient
from geometry_msgs.msg import PointStamped
import numpy as np
import copy
import requests
from flask import Flask, Response, request, jsonify
from urlparse import urlparse
import threading
import signal

class GlipRequest:
    def __init__(self):
        self.image_front = None
        self.image_left = None
        self.image_right = None

        # Set up the ROS subscribers
        cam_front_topic = rospy.get_param('~cam_front','image')
        cam_left_topic = rospy.get_param('~cam_left','image')
        cam_right_topic = rospy.get_param('~cam_right','image')

        self.image_front_subscriber = rospy.Subscriber(cam_front_topic, CompressedImage, self.image_front_callback)
        self.image_left_subscriber = rospy.Subscriber(cam_left_topic, CompressedImage, self.image_left_callback)
        self.image_right_subscriber = rospy.Subscriber(cam_right_topic, CompressedImage, self.image_right_callback)

        # Set up the ROS publishers
        self.center_point_pub = rospy.Publisher('center_point', PointStamped, queue_size=10)
        self.bounding_box_pub = rospy.Publisher('bounding_box_image/image/compressed', CompressedImage, queue_size=10)
        self.concate_image_pub = rospy.Publisher('concate_image/image/compressed', CompressedImage, queue_size=10)
        self.full_message_pub = rospy.Publisher('full_message', NavClient, queue_size=10)

        self.bridge = CvBridge()

        self.server_url = rospy.get_param('~server_url','http://localhost:5000')
        self.host_url = rospy.get_param('~host_url','http://localhost:5000')
        self.robot_id = rospy.get_param('~robot_id', 'test')
        self.init_flask()
        self.init_connection()

        # Catch CTRL-C to terminate threads
        signal.signal(signal.SIGINT, self.exit_node)

    def init_flask(self):
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024  # 40 MB
        self.app.add_url_rule('/get_image', view_func=self.send_image, methods=['GET'])
        self.app.add_url_rule('/receive_command', view_func=self.receive_command, methods=['POST'])
        self.create_flask_thread()

    def create_flask_thread(self):
        self.flask_thread = threading.Thread(target=self.run_flask)
        self.flask_thread.daemon = True
        self.flask_thread.start()

    def run_flask(self):
        parsed_url = urlparse(self.host_url)
        host = parsed_url.hostname
        port = parsed_url.port
        rospy.loginfo("Running Flask server on {}:{}".format(host, port))
        self.app.run(debug=False, host=host, port=port, use_reloader=False)

    def init_connection(self):
        data = {'robot_id': self.robot_id}
        response = requests.post(self.server_url + '/connection', json=data)
        if response.status_code == 204:
            rospy.loginfo("Connected to Server")
        else:
            rospy.logwarn("Failed to Connect to Server")

    def exit_node(self, sig, frame):
        rospy.loginfo("Terminating Flask server")
        rospy.signal_shutdown('Node terminated')
        exit(0)

    def image_front_callback(self, msg):
        self.image_front = msg

    def image_left_callback(self, msg):
        self.image_left = msg

    def image_right_callback(self, msg):
        self.image_right = msg

    def create_request_data(self):
        # Convert ROS images to OpenCV images
        cv_image_front = self.bridge.compressed_imgmsg_to_cv2(self.image_front, desired_encoding='rgb8')
        cv_image_left = self.bridge.compressed_imgmsg_to_cv2(self.image_left, desired_encoding='rgb8')
        cv_image_right = self.bridge.compressed_imgmsg_to_cv2(self.image_right, desired_encoding='rgb8')

        # Concatenate images
        cv_single_image = np.concatenate((cv_image_left, cv_image_front, cv_image_right), axis=1)
        en_single_image = cv2.imencode('.jpg', cv_single_image)[1].tobytes()
        return en_single_image

    def send_image(self):
        if self.image_front is not None and self.image_left is not None and self.image_right is not None:
            data = self.create_request_data()
            return Response(data, mimetype='image/jpeg')
        else:
            rospy.logwarn('All cameras not publishing')
            return Response('Image not available', 404)

    def receive_command(self):
        json_data = request.json
        self.parse_result(json_data)
        return '', 204

    def parse_result(self, response):
        if 'x_coords' in response and 'y_coords' in response:
            for x_coords, y_coords in zip(response['x_coords'], response['y_coords']):
                centroid_x = int((x_coords[0] + x_coords[1]) / 2)
                centroid_y = int((y_coords[0] + y_coords[1]) / 2)

                msg = NavClient()
                msg.header.stamp = rospy.Time.now()
                msg.x = centroid_x
                msg.y = centroid_y
                msg.left = int(x_coords[0])
                msg.lower = int(y_coords[0])
                msg.right = int(x_coords[1])
                msg.upper = int(y_coords[1])

                self.full_message_pub.publish(msg)
        else:
            rospy.logwarn("Invalid JSON response")

if __name__ == '__main__':
    rospy.init_node('nav_client')
    nav_client = ViperRequest()
    rospy.spin()

