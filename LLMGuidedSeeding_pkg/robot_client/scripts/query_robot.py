import rospy
from nav_msgs.msg import Odometry

class OdometryListener:
    def __init__(self, topic):
        # Initialize the ROS node
        rospy.init_node('odometry_listener', anonymous=True)

        # Current pose placeholder
        self.current_pose = None

        # Set up the subscriber to the Odometry topic
        self.subscriber = rospy.Subscriber(topic, Odometry, self.callback)

    def callback(self, msg):
        # Update current pose with the new data received
        self.current_pose = msg.pose.pose

    def get_pose(self):
        # Return the current pose
        return self.current_pose


'''
if __name__ == '__main__':
    # Create an instance of OdometryListener
    listener = OdometryListener('/"H03/odometry"')  # replace '/odom' with your actual topic

    # Spin to keep the script from exiting until this node is stopped
    rospy.spin()

    # Example usage of get_pose
    # It's a bit tricky to directly print the pose as rospy.spin() is blocking
    # Typically, the get_pose method would be used within another callback or a different part of your application
'''