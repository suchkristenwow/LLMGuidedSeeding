from flask import Blueprint, jsonify, Response, request, render_template
import json
import cv2 as cv
import numpy as np
import rospy
from sensor_msgs.msg import Image as SensorImage
import time
import threading
from cv_bridge import CvBridge, CvBridgeError
from .transform import CamProjector
from . import socketio

#from LLMGuidedSeeding_pkg.utils.llm_utils import generate_with_openai

app_routes = Blueprint('app_routes', __name__)

# initialize frame and _frame. _frame must be a png so that imencode.tobytes() can be called properly
_frame = open('backend/app/init_callback_img.png', 'rb').read()
frame = None
is_paused = False
bridge = CvBridge()
cv_image = None
paused_frame = None
drawing_frame = None
points = []
drawing = None
bridge = CvBridge()

################### Utility functions for the backend ##########################
def image_callback(data):
    """Hook function to pull out our frames from Rospy.Subscriber"""
    global frame, _frame, cv_image #frame_count
    if frame is not None and _frame is not None:
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        #_frame = cv2.imencode('.jpg', cv2.resize(frame, (400,225)))[1].tobytes()
        _frame = cv.imencode('.jpg',frame)[1].tobytes()
        #_frame = cv2.cvtColor(_frame,cv2.COLOR_BGR2RGB)
    cv_image = bridge.imgmsg_to_cv2(data)
    frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    #frame_count += 1

def lidar_callback(data):
    return None
    #print("Lidar Data:")
    #print(data.data)
    
def gen():
    """Video streaming generator function."""
    global _frame, paused_frame

    while True:
        # The time delay value should be equal to the 1/FPS value
        time.sleep(0.1)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + _frame + b'\r\n')

# Function to continuously update the OpenCV window
def update_window():
    global cv_image
    while True:
        cv.imshow("Video 1", cv_image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# logic for our callback: Hold down the left click to drop points as you move the mouse
def draw_polylines(event, x, y, flags, param):
    global drawing, drawing_frame,  points # we have to run the global call again in this function
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            # we need to gather points from the user 
            cv.circle(drawing_frame,(x,y),1,(0,0,255),-1)
            points.append([x,y])
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
    # if we want the dots to persist across the frames then we need to move cv.polylines to be inside opening messages loop
    elif event == cv.EVENT_RBUTTONDOWN:  
        if len(points) > 1:
            points_arr = np.array(points, np.int32)
            points_arr = points_arr.reshape((-1, 1, 2))
            cv.polylines(drawing_frame, [points_arr], isClosed=False, color=(0, 255, 0), thickness=3)
        return True
    # clear points
    if event == cv.EVENT_LBUTTONDBLCLK:
        # cv.circle(drawing_frame, (x,y), 3, (0,255,0), -1)
        # points.append([x,y])
        points = []
        drawing_frame = cv.UMat(og_drawing_frame.copy())
            
################## App Routes #####################
@app_routes.route('/backend/image_stream')
def image_stream():
    """Video streaming route. Put this in the html or css that you'd like to display it"""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app_routes.route('/backend/pause')
def pause():
    """Endpoint to serve the paused frame."""
    global paused_frame, _frame
    paused_frame = _frame
    return Response(paused_frame, mimetype='image.jpeg')
    


@app_routes.route('/backend/sketch_boundary')
def drawing():
    global cv_image,og_drawing_frame, drawing_frame, paused_frame, points

    og_drawing_frame = cv_image
    drawing_frame = cv.UMat(og_drawing_frame.copy())

    cv.namedWindow("Video 1", cv.WINDOW_GUI_NORMAL | cv.WINDOW_AUTOSIZE)
    cv.moveWindow("Video 1", 500, 250)
    # Resize the window
    #cv.resizeWindow("Video 1", 800, 600)  # Adjust the size according to your preference
    cv.setMouseCallback('Video 1', draw_polylines)
    
    # Main loop
    while True:
        # Display the window
        cv.imshow("Video 1", drawing_frame)
        
        # Check if the window is closed by the user
        
        # Check for any key pressed, wait for 1 millisecond
        key = cv.waitKey(1) & 0xFF
        # Check if 'x' key is pressed or the window is closed
        if key == ord('x') or key == 27:
            break
    # Clean up
    cv.destroyAllWindows()
    #print(points)
    #return Response(paused_frame, mimetype='multipart/x-mixed-replace; boundary=frame')
    #return Response(drawing_frame, mimetype='image.jpeg')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app_routes.route('/backend/proj_sketch')
def proj_sketch():
    global points
    camera_tf = [0.152758, 0.00, 0.0324, 0,0,0] # [0.152758, 0.00, 0.0324, 0,0,0]
    robot_pose = [0,0,0,0,0,0]

    cam_projector = CamProjector(1, camera_pose=camera_tf, robot_pose=robot_pose)
    sketch_proj = np.array([cam_projector.project(c) for c in points])[:,:3]


@app_routes.route('/backend/player')
def index():
    """Video streaming home page from the Backend with a simple HTML."""
    return render_template('player2.html')

@app_routes.route('/backend/process_message', methods=['POST'])
def process_message():
    data = request.get_json()
    print(data)
    print()

    return jsonify(data)

###################### websocket ###################3
@socketio.on('message')
def handle_message(message):
    print(f'Received message: {message} \n')
    # Process message and send response if needed
    socketio.send(f'Response from server: {message}')