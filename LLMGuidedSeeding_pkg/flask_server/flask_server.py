import rospy
import requests
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Point
from std_msgs.msg import Header
#from robot_client import ArtifactInfo
from flask import Flask, request, jsonify
import cv2
import numpy as np
import threading

# Hypothetical ArtifactInfo message structure. Replace with your actual message.
'''
class ArtifactInfo:
    def __init__(self, header, artifact_id, position, obj_class, obj_prob):
        self.header = header
        self.artifact_id = artifact_id
        self.position = position
        self.obj_class = obj_class
        self.obj_prob = obj_prob
        self.waypoint = None
        self.at_waypoint = False
'''

class CameraServer:
    def __init__(self):
        # Initialize ROS Node
        rospy.init_node('camera_server')

        # Set up subscribers for images
        self.image_front_sub = rospy.Subscriber('camera_front', CompressedImage, self.image_callback, callback_args='front')
        self.image_left_sub = rospy.Subscriber('camera_left', CompressedImage, self.image_callback, callback_args='left')
        self.image_right_sub = rospy.Subscriber('camera_right', CompressedImage, self.image_callback, callback_args='right')

        # Subscribers for posearray_spot and posearray_frontier
        self.posearray_spot_sub = rospy.Subscriber('posearray_spot', PoseArray, self.posearray_callback)
        self.posearray_frontier_sub = rospy.Subscriber('posearray_frontier', PoseArray, self.posearray_frontier_callback)

        # Subscriber for projection_artifacts
        #self.projection_artifacts_sub = rospy.Subscriber('projection_artifacts', ArtifactInfo, self.projection_artifacts_callback)

        # Storage for images, posearrays, and projection artifacts
        self.images = {'front': None, 'left': None, 'right': None}
        self.posearray_spot = None
        self.posearray_frontier = None
        #self.projection_artifacts = []

        # CV Bridge
        self.bridge = CvBridge()

        # Flask app
        self.app = Flask(__name__)
        self.add_endpoints()
        self.flask_thread = threading.Thread(target=self.run_flask)
        self.flask_thread.start()

    def query_near(self):
        """
        Returns the current status of being at the waypoint.
        """
        return jsonify({"at_waypoint": self.at_waypoint})

    # Method to receive and process waypoints
    def receive_waypoint(self):
        try:
            waypoint = request.get_json()
            print("Received waypoint:", waypoint)
            
            # Update the waypoint attribute
            self.waypoint = waypoint

            return jsonify({"message": "Waypoint received successfully"}), 200
        except Exception as e:
            print(f"Error processing waypoint: {e}")
            return jsonify({"error": str(e)}), 500

    def image_callback(self, msg, camera):
        self.images[camera] = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def posearray_callback(self, msg):
        self.posearray_spot = [[pose.position.x, pose.position.y, pose.position.z,
                                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w] for pose in msg.poses]

    def posearray_frontier_callback(self, msg):
        self.posearray_frontier = [[pose.position.x, pose.position.y, pose.position.z,
                                    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w] for pose in msg.poses]

    def projection_artifacts_callback(self, msg):
        '''
        artifact = {
            'header': msg.header.stamp.to_sec(),
            'artifact_id': msg.artifact_id,
            'position': {'x': msg.position.x, 'y': msg.position.y, 'z': msg.position.z},
            'obj_class': msg.obj_class,
            'obj_prob': msg.obj_prob
        }
        '''
        self.projection_artifacts.append(msg)

    def add_endpoints(self):
        self.app.add_url_rule('/image', 'get_image', self.send_image, methods=['GET'])
        self.app.add_url_rule('/posearray_spot', 'get_posearray_spot', self.send_posearray_spot, methods=['GET'])
        self.app.add_url_rule('/posearray_frontier', 'get_posearray_frontier', self.send_posearray_frontier, methods=['GET'])
        #self.app.add_url_rule('/projection_artifacts', 'get_projection_artifacts', self.send_projection_artifacts, methods=['GET'])
        self.app.add_url_rule('/glip_labels', 'receive_glip_labels', self.receive_glip_labels, methods=['POST'])
        self.app.add_url_rule('/query_near', 'query_near', self.query_near, methods=['GET'])
        self.app.add_url_rule('/waypoint', 'receive_waypoint', self.receive_waypoint, methods=['POST'])

    def receive_glip_labels(self):
        try:
            # Extract GLIP label data from the request
            glip_labels = request.get_json()

            # Validate and process the GLIP labels (if necessary)
            # Here you can implement logic to handle the GLIP labels
            print("Received GLIP labels:", glip_labels)
            self.glip_detections = glip_labels

            # Return a success response
            return jsonify({"message": "GLIP labels received successfully"}), 200
        except Exception as e:
            # Handle any exceptions
            print(f"Error processing GLIP labels: {e}")
            return jsonify({"error": str(e)}), 500

    def run_flask(self):
        self.app.run(debug=False, host='0.0.0.0', port=7000)

    def concatenate_images(self):
        images = [self.images[camera] for camera in ['left', 'front', 'right'] if self.images[camera] is not None]
        if images:
            return cv2.hconcat(images)
        return None

    def send_image(self):
        concatenated_image = self.concatenate_images()
        if concatenated_image is not None:
            _, buffer = cv2.imencode('.jpg', concatenated_image)
            return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}
        return 'Images not available', 404

    def query_glip(self):
        concatenated_image = self.concatenate_images()
        if concatenated_image is not None:
            # Encode the image to JPEG format
            success, encoded_image = cv2.imencode('.jpg', concatenated_image)
            if not success:
                print("Error encoding image")
                return

            try:
                # Prepare the headers for the HTTP request
                headers = {'Content-Type': 'image/jpeg'}

                # Send the image to the GLIP server
                response = requests.post('http://localhost:7000/process', data=encoded_image.tobytes(), headers=headers)

                # Check if the request was successful
                if response.status_code == 200:
                    # Parse the response
                    self.glip_response = response.json()
                    print("GLIP response received:", self.glip_response)
                else:
                    print("Error in GLIP server response:", response.status_code)
            except requests.exceptions.RequestException as e:
                print("Error querying GLIP server:", e)
        else:
            print("Concatenated image is not available")

    def send_posearray_spot(self):
        if self.posearray_spot:
            return jsonify(self.posearray_spot)
        return 'Pose array spot not available', 404

    def send_posearray_frontier(self):
        if self.posearray_frontier:
            return jsonify(self.posearray_frontier)
        return 'Pose array frontier not available', 404

    '''
        def send_projection_artifacts(self):
        return jsonify(self.projection_artifacts)
    '''

if __name__ == '__main__':
    server = CameraServer()
