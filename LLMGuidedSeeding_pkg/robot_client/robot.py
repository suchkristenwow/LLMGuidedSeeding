import rospy
from sensor_msgs.msg import Image 
from nav_msgs.msg import Odometry,Path 
from std_msgs.msg import String, Int16  
from geometry_msgs.msg import PoseStamped 
from cv_bridge import CvBridge, CvBridgeError
import numpy as np 
from PIL import Image as PILImage 
import cv2 
import threading 
import tf.transformations as tf
import time 
import octomap
from LLMGuidedSeeding_pkg import *
from collections import deque 

class identified_object: 
    def __init__(self,object_id,init_observation,label,history_size=10): 
        self.mu = init_observation 
        self.sigma = np.eye(2) 
        self.label = label 
        self.object_id = object_id 
        self.visited = False 
        self.observation_history = deque([init_observation], maxlen=history_size)

    def integrate_new_observation(self,observation): 
        self.observation_history.append(observation)
        self.mu = np.median(self.observation_history, axis=0)
        
        z = np.array(observation[:2])  # Observed x, y position

        # Predict step: (In a static case, we don't have a motion model, so the predict step does nothing)
        mu_pred = self.mu
        sigma_pred = self.sigma + np.eye(2) * 0.05  # Add process noise

        # Update step:
        H = np.eye(2)  # Observation model (we directly observe x, y)
        R = np.eye(2) * 0.1  # Observation noise covariance
        y = z - mu_pred  # Innovation: difference between prediction and observation
        S = H @ sigma_pred @ H.T + R  # Innovation covariance
        K = sigma_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

        self.mu = mu_pred + K @ y  # Updated state estimate
        self.sigma = (np.eye(2) - K @ H) @ sigma_pred 

def pixel_to_ray(pixel_coord, camera_intrinsics):
    """Convert a pixel coordinate to a normalized ray in the camera frame."""
    u, v = pixel_coord
    fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
    cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']

    # Convert pixel coordinate to normalized ray
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0  # Assuming depth is 1 unit (ray direction)

    ray = np.array([x, y, z])
    ray /= np.linalg.norm(ray)  # Normalize the ray

    return ray

def transform_ray_to_world(ray, camera_pose):
    """Transform a ray from the camera frame to the world frame using the camera pose."""
    # Extract the rotation (as a matrix) and translation from the camera pose
    rotation_matrix = tf.transformations.quaternion_matrix([camera_pose.orientation.x,
                                                            camera_pose.orientation.y,
                                                            camera_pose.orientation.z,
                                                            camera_pose.orientation.w])[:3, :3]
    translation = np.array([camera_pose.position.x,
                            camera_pose.position.y,
                            camera_pose.position.z])

    # Transform the ray to the world frame
    ray_world = rotation_matrix.dot(ray) + translation

    return ray_world

def ray_cast_octomap(octomap_obj, origin, direction):
    """Perform ray casting in the octomap to find the intersection with occupied space."""
    max_range = 100.0  # Set maximum range for ray casting
    key = octomap_obj.searchRay(origin, direction, max_range)

    if key is not None:
        endpoint = octomap_obj.keyToCoord(key)
        return np.linalg.norm(endpoint - origin)  # Distance from the origin
    else:
        return None  # No intersection found
    
class Robot:
    def __init__(self, config_path, vehicle_prefix='/H03'):
        self.static_transformer = robotTransforms(config_path)
        
        with open(config_path, "r") as f:
            self.settings = toml.load(f)

        self.observation_frequency  = self.settings["robot"]["observation_frequency"]
        self.confidence_threshold = self.settings["robot"]["confidence_threshold"]

        self.ready_to_act = True 
        self.current_waypoint = None
        self.current_pose = None
        self.current_map = {}
        self.current_path = None 
        self.current_odom_msg = None
        self.planter_position = 0 
        self.planted_locations = []

        self.bridge = CvBridge()

        # Initialize the ROS node
        rospy.init_node("robot_control_api", anonymous=True)
        self.task_publisher = rospy.Publisher(vehicle_prefix + "forceTask", String, queue_size=10)
        self.path_publisher = rospy.Publisher(vehicle_prefix + "path_out", Path, queue_size=10)
        self.waypoint_publisher = rospy.Publisher(vehicle_prefix + "/frontier_goal_point", queue_size=10)

        odom_topic = vehicle_prefix + "/odometry"
        rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        # Cameras
        self.camera_topics = {
            "front": vehicle_prefix + "/cam_front",
            "left": vehicle_prefix + "/cam_left",
            "right": vehicle_prefix + "/cam_right"
        }
        self.camera_images = {key: None for key in self.camera_topics.keys()}

        self.down_camera_topics = {
            "left": vehicle_prefix + "/left_down/cam", 
            "right": vehicle_prefix + "/right_down/cam"
        }

        for key, topic in self.camera_topics.items():
            rospy.Subscriber(topic, Image, self.image_callback, callback_args=key)
        
        for key, topic in self.down_camera_topics.items():
            rospy.Subscriber(topic, Image, self.down_image_callback, callback_args=key)  

        rospy.Subscriber("arduino_serial",Int16,self.planter_callback)

        # Initialize YoloWorldInference
        self.yolo_world = YoloWorldInference()

        self.ros_thread = threading.Thread(target=self.ros_spin)
        self.ros_thread.daemon = True
        self.ros_thread.start() 

        # Start the movement monitor thread
        self.movement_monitor_thread = threading.Thread(target=self.monitor_movement)
        self.movement_monitor_thread.daemon = True
        self.movement_monitor_thread.start()

        #MAPPING 
        self.sized_labels = {} 

    def ros_spin(self):
        rospy.spin() 

    def monitor_movement(self):
        """Monitor the robot's movement and call get_current_observations if it has moved."""
        rate = rospy.Rate(self.observation_frequency) 
        while not rospy.is_shutdown():
            if self.has_moved():
                self.get_current_observations()
            rate.sleep()

    def has_moved(self):
        """Check if the robot has moved since the last check."""
        if self.previous_pose is None:
            self.previous_pose = self.get_current_pose()
            return False

        current_pose = self.get_current_pose()
        distance_moved = np.linalg.norm(current_pose[:2] - self.previous_pose[:2])
        heading_changed = abs(current_pose[-1] - self.previous_pose[-1])

        self.previous_pose = current_pose

        # Consider the robot moved if it moved more than 0.01 meters or changed heading more than 0.01 radians
        return distance_moved > 0.01 or heading_changed > 0.01
    
    #CONTROL 
    def follow_path(self):  
        if not self.ready_to_act:
            while not self.ready_to_act: 
                print("cant act right now ... waiting for another task to complete") 
                time.sleep(0.2)  

        self.ready_to_act = False 
        self.path_publisher.publish(self.current_path)

        last_pose_msg = self.current_path.poses[-1]
        twoD_goal_point = np.array([last_pose_msg.pose.position.x,last_pose_msg.pose.position.y])
        d = np.linalg.norm(self.get_current_pose()[:2] - twoD_goal_point) 
        while d > 0.05: 
            d = np.linalg.norm(self.get_current_pose()[:2] - twoD_goal_point)  
            time.sleep(0.5)
        self.ready_to_act = True  

    def go_to_waypoint(self): 
        if not self.ready_to_act:
            while not self.ready_to_act: 
                print("cant act right now ... waiting for another task to complete") 
                time.sleep(0.2) 
        
        msg = PoseStamped()
        #TO DO: HEADER? 
        msg.pose = self.current_waypoint
        self.waypoint_publisher.publish(msg)

        #change ready to act after we have reached the way point 
        twoD_goal_point = np.array([self.current_waypoint.pose.position.x, self.current_waypoint.pose.position.y]) 
        d = np.linalg.norm(self.get_current_pose()[:2] - twoD_goal_point) 
        while d > 0.05: 
            d = np.linalg.norm(self.get_current_pose()[:2] - twoD_goal_point)  
            time.sleep(0.5)
        self.ready_to_act = True 

    def explore_using_volumetric_gain(self):
        msg = String()
        msg.data = "Explore"
        self.task_publisher.publish(msg)

    def hold_up(self):
        '''
        STOP MOVING 
        '''
        self.ready_to_act = False 
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.poses = []
        self.path_publisher.publish(msg)

    def plant(self):  
        '''Returns the planted coordinate'''
        if not self.ready_to_act:
            while not self.ready_to_act: 
                print("cant act right now ... waiting for another task to complete") 
                time.sleep(0.2)  
        
        #Make sure we are not moving when we plant 
        self.hold_up()

        msg = String()
        msg.data = "Plant"
        self.ready_to_act = False 
        self.task_publisher.publish(msg) 
        planted_coord = self.static_transformer.get_planter_position(self.get_current_pose())
        while self.planter_position != 0:
            time.sleep(0.2) 
        self.ready_to_act = True 

        self.planted_locations.append(planted_coord)

        return planted_coord
    
    def planter_callback(self,msg): 
        self.planter_position = msg.data
 
    #ODOMETRY CLASSES 
    def get_current_pose(self): 
        '''Return x,y,yaw as np.ndarray of size 1x3 of the current pose''' 
        x = self.current_odom_msg.pose.pose.position.x 
        y = self.current_odom_msg.pose.pose.position.y 
        z = self.current_odom_msg.pose.pose.position.z  

        # Extract orientation (quaternion)
        quaternion = (
            self.current_odom_msg.pose.pose.orientation.x,
            self.current_odom_msg.pose.pose.orientation.y,
            self.current_odom_msg.pose.pose.orientation.z,
            self.current_odom_msg.pose.pose.orientation.w
        )

        # Convert quaternion to roll, pitch, yaw
        roll, pitch, yaw = tf.euler_from_quaternion(quaternion) 
        return np.array([x,y,z,roll,pitch,yaw])

    def odometry_callback(self, msg):
        """Handle odometry messages."""
        self.current_odom_msg = msg
    
    #IMAGE DETECTIONS 
    def check_environment_for_something(self,thing): 
        if thing in self.current_map.keys():
            things = self.current_map[thing]

            thing_locations = np.array([x.mu for x in things]); thing_locations = np.reshape(thing_locations,(len(things),2))
            #return the one thats closest
            distances = np.sum((thing_locations - self.current_map[thing][:2]) ** 2, axis=1)
            
            # Find the index of the smallest distance
            closest_index = np.argmin(distances)
            
            # Return the location with the smallest distance
            return thing_locations[closest_index]
        
        else:
            return None 

    def get_current_observations(self): 
        results = {}
        for camera in self.camera_topics.keys():
            results[camera] = self.process_frame(camera) 

        self.update_map(results) 

    def image_callback(self, msg, camera_name):
        """Handle image messages from cameras."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.camera_images[camera_name] = cv_image
        except CvBridgeError as e:
            rospy.logerr(f"Error converting image from {camera_name} camera: {e}")

    def capture_image(self, camera_name):
        """Get the latest image from the specified camera."""
        return self.camera_images[camera_name]

    def identify_objects(self, image):
        """Use YoloWorldInference to identify objects in the image."""
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results = self.yolo_world.infer(pil_image)

        # Extract bounding boxes, scores, and class names
        boxes = []
        scores = []
        names = []
        for result in results:
            for box in result.boxes:
                boxes.append(box.xyxy.tolist())
                scores.append(box.conf.tolist())
                names.append(result.names[int(box.cls.tolist())])

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": names
        }

    def process_frame(self, camera_name):
        """Capture an image from the specified camera, send it for identification, and process the results."""
        # Capture image
        image = self.capture_image(camera_name)

        if image is None:
            rospy.logwarn(f"No image available from {camera_name} camera.")
            return None

        # Identify objects in the image
        results = self.identify_objects(image)

        # Process and use the results (e.g., logging, decision-making)
        print(f"Identified {len(results['labels'])} objects from {camera_name} camera: {results['labels']}")
        return results 

    # MAPPING 
    def get_depth_px_coord(self,pixel_coord,camera_name): 
        '''Query the octomap, ray cast projection to get estimated depth'''
        robot_pose = self.get_current_pose() 

        if "front" in camera_name: 
            camera_pose = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.front_camera_tf))  
        elif "left" in camera_name: 
            camera_pose = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.left_camera_tf))  
        elif "right" in camera_name: 
            camera_pose = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.right_camera_tf))  

        # Convert the pixel to a 3D ray in the camera frame
        ray_camera = pixel_to_ray(pixel_coord, self.camera_intrinsics[camera_name])

        # Transform the ray to the world frame
        ray_world = transform_ray_to_world(ray_camera, camera_pose)

        # Convert the Octomap message to an Octomap object
        octomap_obj = octomap.OcTree(octomap_msg.data)

        # Perform ray casting
        origin = np.array([camera_pose.position.x,
                        camera_pose.position.y,
                        camera_pose.position.z])
        depth = ray_cast_octomap(octomap_obj, origin, ray_world)

        return depth

    def update_map(self,results):   
        """
        Update the robot's map with new observations.
        
        Args:
            results (dict): A dictionary where keys are camera names and values are lists of detected objects,
                            each containing "boxes", "scores", and "labels".
        """
        for camera_name, detection in results.items():
            for i in range(len(detection['labels'])):
                label = detection['labels'][i]
                box = detection['boxes'][i]  # Assuming bounding box is [x_min, y_min, x_max, y_max]
                score = detection['scores'][i]

                if score < self.confidence_threshold:  
                    continue 

                # Convert bounding box to object center (assuming x, y are the center)
                observation_px_coord = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

                depth = self.get_depth_px_coord(observation_px_coord,camera_name) 

                camera_topic = self.camera_topics[camera_name]

                robot_pose = self.get_current_pose() 

                if "front" in camera_name: 
                    camera_pose = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.front_camera_tf))  
                elif "left" in camera_name: 
                    camera_pose = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.left_camera_tf))  
                elif "right" in camera_name: 
                    camera_pose = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.right_camera_tf))  

                object_center = CamProjector(depth,camera_topic,camera_pose,robot_pose) 

                # If this label is already in the map, update the nearest object
                if label in self.current_map:
                    if not label in self.sized_labels: 
                        label_size = self.get_label_size(label) 
                    else:
                        label_size = self.sized_labels[label] 

                    existing_objects = self.current_map[label]

                    # Find the closest existing object using Euclidean distance
                    distances = [np.linalg.norm(obj.mu - object_center) for obj in existing_objects]
                    min_dist = min(distances)

                    if min_dist < label_size * 1.25:  # Threshold for association (1.0 meter in this example)
                        nearest_object = existing_objects[distances.index(min_dist)]
                        nearest_object.integrate_new_observation(object_center)
                    else:
                        # If no nearby object found, create a new one
                        #object_id,init_observation,label
                        id_ = max([x.object_id for x in existing_objects]) + 1
                        new_object = identified_object(id_, object_center, label)
                        self.current_map[label].append(new_object)
                else:
                    # If this label is not in the map, create a new entry
                    #object_id,init_observation,label
                    self.current_map[label] = [identified_object(0, object_center, label)]