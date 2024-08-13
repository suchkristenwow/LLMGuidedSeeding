import rospy
from robot_client.robot import Robot


if __name__ == '__main__':
    rospy.init_node('robot_node')
    rate = rospy.Rate(30)  # Set the desired frequency (30Hz)

    seeder = Robot()
    seeder.create_flask_thread()
    
    while not rospy.is_shutdown():
        # Perform object detection tasks here
        seeder.run()
        rate.sleep()
