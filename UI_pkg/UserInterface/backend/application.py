import threading
import rospy
import cv2
import numpy as np
from app import create_app
from sensor_msgs.msg import Image as SensorImage
import time

application = create_app()



if __name__ == "__main__":

    #app.run(host='0.0.0.0', debug=True)
    application.run(debug = True)
