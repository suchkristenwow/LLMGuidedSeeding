import numpy as np
import image_geometry
import sensor_msgs
from geometry_msgs.msg import Point

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
# import this library so that we can use plot_points ax = fig.add_subplot(111, projection='3d') 
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.transforms import Affine2D
import cv2 as cv
import rosbag
from cv_bridge import CvBridge
# from main import sketch_boundary

class CamProjector:

    def __init__(self, depth, camera_pose, robot_pose) -> None:
        self.cam_model = CamProjector.get_camera_model()
        self.camera_pose = camera_pose
        self.robot_pose = robot_pose
        self.depth = depth

    @staticmethod
    def pose_to_transformation_matrix(pose):
        tf_matrix = np.zeros((4,4))
        r = R.from_euler("XYZ", pose[3:], degrees=False)
        tf_matrix[:3,:3] = r.as_matrix()
        tf_matrix[0,3] = pose[0]
        tf_matrix[1,3] = pose[1]
        tf_matrix[3,3] = pose[2]
        tf_matrix[3,3] = 1 
        return tf_matrix

    def project_pixel(self, pixel):
        ray = np.asarray(self.cam_model.projectPixelTo3dRay(pixel))
        # Convert to Point
        point = ray * self.depth
        return point
    
    def convert_optical_to_nav(self, cam_point):
        cam_nav_frame_point = Point()
        cam_nav_frame_point.x = cam_point[2]
        cam_nav_frame_point.y = -1.0 * cam_point[0]
        cam_nav_frame_point.z = -1.0 * cam_point[1]
        return cam_nav_frame_point

    
    def apply_cam_transformation(self, point):
        cam_tf = self.pose_to_transformation_matrix(self.camera_pose)
        robot_tf = self.pose_to_transformation_matrix(self.robot_pose)
        # First apply cam_tf then robot_tf
        full_tf = np.dot(robot_tf, cam_tf)
        # Replace numpify -> point is a geometry_msgs.msg Point object, which has x, y, z
        point_np = np.append(np.array([point.x, point.y, point.z]),1)
        new_point = np.dot(full_tf, point_np)
        return new_point


    def project(self,pixel):
        # Project To A Point in Camera frame
        cam_point = self.project_pixel(pixel)
        # Camera Point To Cam Nav Frame
        cam_point_frame = self.convert_optical_to_nav(cam_point)
        # Transfrom point
        new_point = self.apply_cam_transformation(cam_point_frame)
        return new_point


    @staticmethod
    def get_camera_model():
        # https://docs.ros.org/en/melodic/api/image_geometry/html/c++/classimage__geometry_1_1PinholeCameraModel.html#a9f7f4cfc429df4868af4f46fa05139d6
        # https://docs.ros.org/en/melodic/api/image_geometry/html/c++/pinhole__camera__model_8h_source.html#l00266
        camera_model = image_geometry.PinholeCameraModel()
        # D,K,R,P describ
        # Down facing Camera:
        # D = [-0.054073840379714966, 0.06357865780591965, 0.0001570347958477214, 0.0004054234887007624, -0.019391480833292007]
        # K =  [382.0611572265625, 0.0, 323.7216796875, 0.0, 381.68243408203125, 245.92823791503906, 0.0, 0.0, 1.0]
        # R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # P = [382.0611572265625, 0.0, 323.7216796875, 0.0, 0.0, 381.68243408203125, 245.92823791503906, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        # Front Facing Camera
        D = [-0.25861, 0.06925, 0.001231, 0.001145, 0.0]
        K = [510.1259, 0.0, 420.6644, 0.0, 509.4401, 306.44069, 0.0, 0.0, 1.0]
        R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        P = [417.10114, 0.0, 429.82936, 0.0, 0.0, 458.0275, 308.05336, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        camera_model.D =  D 
        camera_model.K = np.reshape(np.array(K),(3,3))
        camera_model.R = np.reshape(np.array(R),(3,3))
        camera_model.P = np.reshape(np.array(P),(3,4))
        camera_model.width = 608
        camera_model.height = 808 
        camera_model.binning_x = 0
        camera_model.binning_y = 0
        msg_roi = sensor_msgs.msg.RegionOfInterest()
        msg_roi.x_offset = 0
        msg_roi.y_offset = 0
        msg_roi.height = 0
        msg_roi.width = 0
        msg_roi.do_rectify = False 
        camera_model.raw_roi = msg_roi
        camera_model.stamp = 0
        return camera_model

    @staticmethod
    def plot_points(points, camera_point):
        # Extract x, y, and z coordinates from the points
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        z_coords = [point[2] for point in points]
        
        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.axis('equal')
        ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')
        # what is the z value of the camera
        ax.scatter(camera_point[0], camera_point[1], camera_point[2], c = 'g', marker = 'o')
        
        # Set labels for each axis
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        # Set plot title
        ax.set_title('3D Scatter Plot of Points')
        
        # print the points
        # print(points)
        # Show the plot
        plt.show()
        


    @staticmethod
    def plot_x_y_points_robot(points, robot_pose, camera_pose):
        pointer_x = robot_pose[0] + 0.25 * np.cos(robot_pose[5])
        pointer_y = robot_pose[1] + 0.25 * np.sin(robot_pose[5])

        # Extract x and y coordinates from the points
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        # Create a new figure and axis
        fig, ax = plt.subplots()

        # Plot the robot's pose and pointer
        ax.scatter(robot_pose[0], robot_pose[1], c="red")
        ax.scatter(camera_pose[0], camera_pose[1], c="green")
        ax.text(camera_pose[0], camera_pose[1], f'({camera_pose[0]},{camera_pose[1]})', fontsize=12, color='green')
        ax.plot([robot_pose[0], pointer_x], [robot_pose[1], pointer_y], c="red")

        # Plot the points
        ax.scatter(x_coords, y_coords, c='blue', marker='o')
        ax.text(x_coords[0], y_coords[0], f'({x_coords[0],y_coords[0]})', fontsize=12, color='green' )

        # Add rectangle around the robot
        width = 0.99
        height = 0.67
        rectangle = patches.Rectangle((robot_pose[0] - width, robot_pose[1] - height/2), width, height, linewidth=1, edgecolor='green', facecolor='none')
        rotation_matrix = Affine2D().rotate_around(robot_pose[0], robot_pose[1], robot_pose[5])
        rectangle.set_transform(rotation_matrix + ax.transData)
        ax.add_patch(rectangle)
        ax.add_patch(rectangle)

        # Set labels for x and y axes
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

        # Show the plot
        plt.show()
