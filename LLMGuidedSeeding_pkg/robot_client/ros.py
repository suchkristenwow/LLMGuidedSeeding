import rospy 
import roslaunch 
from nav_msgs.msg import OccupancyGrid,Odometry,Path  
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose, Point, Quaternion, PointStamped, PoseWithCovariance, TwistWithCovariance
import numpy as np 
from shapely import Polygon 
import math 
import threading 

from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(roll, pitch, yaw):
    # Create a rotation object from Euler angles
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    
    # Convert the rotation to a quaternion (x, y, z, w)
    quaternion = r.as_quat()
    return quaternion

def launch_ros_node(node_name,ros_package,launch_file):
    # Initialize a ROS node (optional if you don't already have one)
    rospy.init_node(node_name, anonymous=True)

    # Create the roslaunch object
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    # Specify the package and launch file to run
    #rrt_exploration,ros_multi_tb3 
    launch_file = roslaunch.rlutil.resolve_launch_arguments([ros_package, launch_file])[0]

    # Create a ROSLaunch object
    launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_file])

    # Start the launch process
    launch.start()

    rospy.loginfo("Launch started")

    # Keep running until the script is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Shutdown the launch process
        launch.shutdown() 

class rosPublisher: 
    def __init__(self,settings,plot_bounds,init_pose,robot_length):  
        self.rrt_planner_node = launch_ros_node("rrt_planner","ros_multi_tb3","single_robot.launch")
        self.exploration_node = launch_ros_node("rrt_exploration","rrt_exploration","single_robot.launch")
        self.map_topic = settings["simulation_parameters"]['simBot_map_topic']
        self.goals_topic = settings["simulation_parameters"]['goal_point'] 
        self.rrt_planner_topic = settings["simulation_parameters"]["path_topic"]
        self.odom_pub = rospy.Publisher(self.odom_topic,Odometry,queue_size=10) 
        self.map_pub = rospy.Publisher(self.map_topic,PointCloud2,queue_size=10)
        self.goal_point_pub = rospy.Publisher(self.goals_topic,PointStamped,queue_size=10) 
        self.map_resolution = 0.15
        self.grid_width = None; self.grid_height = None
        self.grid_origin = None 
        self.occupancy_map_data = self.init_map(plot_bounds,init_pose,robot_length) 
        self.start_map_publishing_thread() 
        self.start_odom_publishing_thread() 
        self.start_cmd_vel_subscriber_thread() 
        self.current_path = None 

    def start_cmd_vel_subscriber_thread(self):
        thread = threading.Thread(target=self.cmd_vel_callback) 
        thread.daemon = True  
        thread.start() 

    def cmd_vel_callback(self,msg):
        self.estimate_pose(msg)
    
    def start_odom_publishing_thread(self):
        # Start a new thread to run the occupancy grid publisher
        thread = threading.Thread(target=self.publish_odometry())
        thread.daemon = True  # Daemonize the thread to exit when the main program exits
        thread.start()  

    # Function to run the ROS publisher in a separate thread
    def start_map_publishing_thread(self):
        # Start a new thread to run the occupancy grid publisher
        thread = threading.Thread(target=self.publish_occupancy_grid())
        thread.daemon = True  # Daemonize the thread to exit when the main program exits
        thread.start() 

    def estimate_pose(self,msg):
        """
        """                

    def publish_occupancy_grid(self):
        """
        Want to publish self.occupancy_map_data as a point cloud
        """
        rate = rospy.Rate(20)
        generated_cloud = PointCloud2() 
        
        while not rospy.is_shutdown():
            generated_cloud.header.stamp = rospy.Time.now() 
            self.map_pub.publish(generated_cloud) 
            rate.sleep()  

    def init_map(self,plot_bounds,init_pose): 
        """
        Init map occupancy 
        """
        delta_x = 1.1*max([max(plot_bounds[:,0]),init_pose[0]]) - min([min(plot_bounds[:,0]),init_pose[0]]) 
        delta_y = 1.1*max([max(plot_bounds[:,1]),init_pose[1]]) - min([min(plot_bounds[:,1]),init_pose[1]])   
        self.grid_origin = Pose(Point(min(delta_x),min(delta_y),0.0),Quaternion(0.0, 0.0, 0.0, 1.0)) 
        self.occupancy_map_data = [-1]*(delta_x,delta_y) 
        self.grid_width = delta_x 
        self.grid_height = delta_y 

    def get_occupancy_grid_idx(self,x_world,y_world):
        # Convert world coordinates to grid cell indices
        i = math.floor((x_world - self.grid_origin.x) / self.map_resolution)
        j = math.floor((y_world - self.grid_origin.y) / self.map_resolution)

        # Check if the indices are within the bounds of the grid
        if i < 0 or i >= self.grid_width or j < 0 or j >= self.grid_height:
            raise ValueError("Coordinates are out of bounds")
        msg_idx = j * self.grid_width + i 
        return msg_idx 

    def update_occupancy_map(self,semantic_map,observed_areas): 
        """
        semantic map is a dictionary with label keys and entries, values are lists of identified objects 
        and publish occupancyGrid message 
        """
        for obstacle_type in semantic_map.keys():
            for obstacle in semantic_map[obstacle_type]: 
                obstacle_location = obstacle.mu 
                msg_idx = self.get_occupancy_grid_idx(obstacle_location[0],obstacle_location[1]) 
                self.occupancy_map_data[msg_idx] = 100 
                obstacle_shape = obstacle.shape 
                obstacle_perimeter_x, obstacle_perimeter_y = obstacle_shape.exterior.xy
                for pt in zip(obstacle_perimeter_x,obstacle_perimeter_y):
                    pt_x,pt_y = pt 
                    msg_idx = self.get_occupancy_grid_idx(pt_x,pt_y)
                    self.occupancy_map_data[msg_idx] = 100

        for area in observed_areas: 
            #x, y = obs.exterior.xy  
            msg_idx = self.get_occupancy_grid_idx(area.centroid[0],area.centroid[1])
            if self.occupancy_map_data[msg_idx] != 100:
                self.occupancy_map_data[msg_idx] = 0 
            for pt in area.exterior.xy: 
                msg_idx = self.get_occupancy_grid_idx(pt[0],pt[1]) 
                if self.occupancy_map_data[msg_idx] != 100:
                    self.occupancy_map_data[msg_idx] = 0 

    def get_path(self,target): 
        """
        Get path from rrt planner given target 
        """
        
    def update_pose(self,current_pose):
        #Publish Odometry 
        msg = Odometry() 
        msg.header = Header() 
        msg.header.frame_id = self.odom_frame 
        msg.header.stamp = rospy.Time.now()
        msg.child_frame_id = self.odom_frame #idk what this is supposed to be :) 
        pose_msg = PoseWithCovariance()  
        pose_msg.pose.position.x = current_pose[0]
        pose_msg.pose.position.y = current_pose[1] 
        pose_msg.pose.position.z = current_pose[2] 
        rot_x,rot_y,rot_z,w = euler_to_quaternion(current_pose[3],current_pose[4],current_pose[5])
        pose_msg.pose.orientation.x = rot_x 
        pose_msg.pose.orientation.y = rot_y 
        pose_msg.pose.orientation.z = rot_z 
        pose_msg.pose.orientation.w = w 
        msg.pose = pose_msg 
        self.odom_pub.publish(msg)