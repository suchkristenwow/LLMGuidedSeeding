import rospy
from robot_client.object_detection_client import ObjectDetectionClient
from robot_client.object_detection_yolo_client import ObjectDetectionClient as YOLOObjectDetectionClient


if __name__ == '__main__':
    rospy.init_node('object_detection_client_node')
    rate = rospy.Rate(20)  # Set the desired frequency (30Hz)

    detector_name = rospy.get_param('detector_name', "yolo")
    od_client = None
    if detector_name == "yolo":
        od_client = YOLOObjectDetectionClient()
    else:
        od_client = ObjectDetectionClient()
    
    #rospy.spin()
    while not rospy.is_shutdown():
        # Perform object detection tasks here
        rate.sleep()
