import os
import numpy as np
import cv2
import time
import rospy
import threading
from flask import Flask, render_template, Response, redirect
from sensor_msgs.msg import Image as SensorImage

frame_count = 0
frame = None

# Add a file path of a blank image here, in order to avoid this variable being empty at the beginning

_frame = open('backend/app/init_callback_img.png', 'rb').read()

app = Flask(__name__)

def image_callback(data):

    global frame, frame_count, _frame

    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #_frame = cv2.imencode('.jpg', cv2.resize(frame, (400,225)))[1].tobytes()
        _frame = cv2.imencode('.jpg',frame)[1].tobytes()
        #_frame = cv2.cvtColor(_frame,cv2.COLOR_BGR2RGB)

    frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    frame_count += 1

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('player.html')

def gen():
    global _frame
    """Video streaming generator function."""
    while True:
        # The time delay value should be equal to the 1/FPS value
        time.sleep(0.1)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + _frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    rospy.Subscriber("/H03/cam_front/image_color", SensorImage, image_callback)
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #This should be done to avoid the flask server conflicting with the node
    threading.Thread(target=lambda: rospy.init_node('visualizer',anonymous=True, disable_signals=True)).start()
    #rospy.Subscriber("/front_camera/color/image_raw", SensorImage, image_callback)
    

    app.run(host='0.0.0.0', debug=True)