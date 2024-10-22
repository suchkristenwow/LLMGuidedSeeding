import threading
import rospy
import cv2
import numpy as np
from app import create_app
from sensor_msgs.msg import Image as SensorImage
import time
import argparse
import logging
import os
from logging.handlers import RotatingFileHandler


application = create_app()



if __name__ == "__main__":
     # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Flask application')
    parser.add_argument('--logging_file', dest='logging_file', type=str,
                        help='Directory for logging output')
    args = parser.parse_args()

    # Use the logging directory if provided
    if args.logging_file:
        # Configure logging
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

        # Create a file handler and set the log level
        file_handler = RotatingFileHandler(args.logging_file, maxBytes=1024*1024, backupCount=10)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))

        # Add the file handler to the root logger
        logging.getLogger().addHandler(file_handler)

        # Set propagate to False to prevent messages from being printed to the terminal
        logging.getLogger().propagate = False
    #app.run(host='0.0.0.0', debug=True)
    application.run(host='0.0.0.0', port=7000, debug = True)
