#!/usr/bin/env python
import rospy
import math
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from octomap_msgs.msg import Octomap

class VelocityPublisher:
    def __init__(self):
        rospy.init_node('velocity_publisher', anonymous=True)
        self.map_recived = False
        self.odom_topic = rospy.get_param('~odom_topic', 'odom')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', 'cmd_vel')
        self.map_topic = rospy.get_param('~octomap_topic', 'map')
        self.ready_topic = rospy.get_param('~ready_topic', 'ready')

        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        self.map_sub = rospy.Subscriber(self.map_topic, Octomap, self.map_callback  )
        self.vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.rady_pub = rospy.Publisher('ready', Bool, queue_size=1, latch=True)

        self.start_x = None
        self.start_y = None
        self.velocity = rospy.get_param('~velocity', 0.5)
        self.distance = rospy.get_param('~start_distance', 5.0)

    def map_callback(self, data):
        self.map_recived = True
        self.map_sub.unregister()
        

    def odom_callback(self, data):
        if not self.map_recived:
            return

        if self.start_x is None or self.start_y is None:
            self.start_x = data.pose.pose.position.x
            self.start_y = data.pose.pose.position.y

        current_x = data.pose.pose.position.x
        current_y = data.pose.pose.position.y
        dist_moved = math.sqrt((current_x - self.start_x) ** 2 + (current_y - self.start_y) ** 2)

        if dist_moved < self.distance:
            self.publish_velocity()
        else:
            self.stop_moving()

    def publish_velocity(self):
        vel_msg = Twist()
        vel_msg.linear.x = self.velocity
        self.vel_pub.publish(vel_msg)

    def stop_moving(self):
        vel_msg = Twist()
        self.vel_pub.publish(vel_msg)
        rospy.loginfo("Reached the goal")
        self.odom_sub.unregister()
        self.publish_ready()
            
    def publish_ready(self):
        ready_msg = Bool()
        ready_msg.data = True
        self.rady_pub.publish(ready_msg)

if __name__ == '__main__':
    try:
        node = VelocityPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
