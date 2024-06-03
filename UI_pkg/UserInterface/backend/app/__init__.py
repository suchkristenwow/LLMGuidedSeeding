from flask import Flask
from flask_cors import CORS # A Flask extension for handling Cross Origin Resource Sharing
import rospy
from sensor_msgs.msg import Image as SensorImage
from sensor_msgs.msg import PointCloud2 
import threading



def create_app():
    application = Flask(__name__)
    CORS(application)
    from .views import app_routes, image_callback, lidar_callback
    application.register_blueprint(app_routes)

    threading.Thread(target=lambda: rospy.init_node('visualizer',anonymous=True, disable_signals=True)).start()
    rospy.Subscriber("/H03/cam_front/image_color", SensorImage, image_callback)
    rospy.Subscriber("/H03/cam_front/image_color", PointCloud2, lidar_callback)
                                                                                                    
    return application