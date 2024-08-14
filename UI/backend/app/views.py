from flask import Blueprint, jsonify, Response, request, render_template
import json
import cv2 as cv
import numpy as np
import rospy
from sensor_msgs.msg import Image as SensorImage
from sensor_msgs.msg import PointCloud2

import sensor_msgs.point_cloud2 as pc2
import time
import threading
from cv_bridge import CvBridge, CvBridgeError
from .transform import CamProjector
import open3d as o3d
import csv
import logging
from . import socketio
import os 
#from LLMGuidedSeeding_pkg.utils.llm_utils import generate_with_openai

app_routes = Blueprint('app_routes', __name__)

# initialize frame and _frame. _frame must be a png so that imencode.tobytes() can be called properly
_frame, frame, paused_frame = open(os.path.abspath(__file__), 'rb').read(), None, None
cv_image, paused_cv_image = None, None
pc_msg, paused_pc_msg = None, None

is_paused = False
drawing_frame = None
points = []
drawing = None
bridge = CvBridge()


################### Utility functions for the backend ##########################
def image_callback(data) -> None:
    """Hook function to pull out our frames from Rospy.Subscriber"""
    global frame, _frame, cv_image 
    if frame is not None and _frame is not None:
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        #_frame = cv2.imencode('.jpg', cv2.resize(frame, (400,225)))[1].tobytes()
        _frame = cv.imencode('.jpg',frame)[1].tobytes()
        #_frame = cv2.cvtColor(_frame,cv2.COLOR_BGR2RGB)
    cv_image = bridge.imgmsg_to_cv2(data)
    frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
 
def lidar_callback(data):
    """Hook function to pull out out LiDAR from Rospy.Subscriber"""
    global pc_msg
    pc_msg = data
   
def gen():
    """Video streaming generator function."""
    global _frame
    while True:
        # The time delay value should be equal to the 1/FPS value
        time.sleep(0.1)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + _frame + b'\r\n')

# def update_window():
#     """Continously update the OpenCV window"""
#     global cv_image
#     while True:
#         cv.imshow("Video 1", cv_image)
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break

def draw_polylines(event, x, y, flags, param):
    """logic for our callback: Hold down the left click to drop points as you move the mouse"""
    global drawing, paused_cv_image, drawing_frame, points # we have to run the global call again in this function
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
        drawing_frame = cv.UMat(paused_cv_image.copy())

def project_sketch(sketch_points: list, paused_pc_message: PointCloud2) -> o3d.geometry.PointCloud:
    """
    Project pixel coordinates into the egocentric lidar when the sketch was made
    Args:
        points (np.ndarray): The sketched pixel coordinates to be processed as a NumPy array.
        paused_pc_msg (pc2.PointCloud2): The environments point cloud data collected in the function pause..
    Returns:
        projected_pcd: (o3d.geometry.PointCloud)
    """
    cam_projector = CamProjector(1, camera_pose = [0.152758, 0.00, 0.0324, 0,0,0], robot_pose = [0,0,0,0,0,0]) # scaling factor of 1
    sketch_proj = np.array([cam_projector.project(c) for c in sketch_points])[:,:3]
    
    generator = pc2.read_points(paused_pc_message, skip_nans = True, field_names = ("x", "y","z")) # iterator object
    msg_points = np.array([point for point in generator])

    # create point clouds -> Points are the msg_points. Since we are only writing to a csv we do not need color
    world_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(msg_points)) # , colors=o3d.utility.Vector3dVector(np.full((msg_points.shape[0], 3), 0.5))
    sketch_pcd = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(sketch_proj)) # , colors = o3d.utility.Vector3dVector(np.full((sketch_proj.shape[0], 3), 0.5))

    # Origin point in point cloud
    origin_pcd = o3d.geometry.PointCloud(points =o3d.utility.Vector3dVector(np.array([[0,0,0]])))
    
    # Project world points to the unit sphere
    world_distances = np.array(world_pcd.compute_point_cloud_distance(origin_pcd)) + .00001
    world_unit = np.asarray(world_pcd.points) / world_distances[:,None] 
 
    # Project sketch points to the unit sphere
    sketch_distances = np.array(sketch_pcd.compute_point_cloud_distance(origin_pcd)) + .0001
    sketch_unit = np.asarray(sketch_pcd.points) / sketch_distances[:,None]

    # Calculate the index of the closest world point to each sketch point
    pcd_distances = np.sqrt(np.sum((world_unit - sketch_unit[:,np.newaxis,:])  ** 2, axis=2))
    closest_ind = np.argmin(pcd_distances, axis = 1)

    # Select the closest world point
    return o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(np.array(world_pcd.points)[closest_ind]))

def display_window():
    global drawing_frame
    cv.namedWindow("Video 1", cv.WINDOW_GUI_NORMAL | cv.WINDOW_AUTOSIZE)
    cv.moveWindow("Video 1", 500, 250)
    cv.setMouseCallback('Video 1', draw_polylines)
    while True:
        cv.imshow("Video 1", drawing_frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('x') or key == 27:
            break
    cv.destroyAllWindows()   

def write_pcd_to_csv(pcd: o3d.geometry.PointCloud, filename: str) -> None:
    """
     Write the incoming point cloud points to a csv 
    Args:
        pcd (o3d.geometry.PointCloud): point cloud data to be saved
        filename (str): name of the file. It should end with .csv
    Returns:
        None: This function does not return any value, but it saves the pcd as a csv file.
    """
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for point in pcd.points:
            csvwriter.writerow(point)

################## App Routes #####################
@app_routes.route('/test')
def test():
    logging.debug("Test endpoint called debug")
    
    print('test endpoint called print')
    return "Test successful", 200

@app_routes.route('/')
def home():
    """Video streaming home page from the Backend with a simple HTML."""
    return render_template('home.html')


@app_routes.route('/backend/image_stream')
def image_stream():
    """Video streaming route. Put this in the html or css that you'd like to display it"""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app_routes.route('/backend/pause')
def pause():
    """Endpoint to serve the paused frame."""
    global paused_frame, _frame, paused_pc_msg, pc_msg, paused_cv_image, cv_image
    paused_pc_msg = pc_msg
    paused_frame = _frame
    paused_cv_image = cv_image
    np.save("UI/still_frame.npy", paused_cv_image)
    return Response(paused_frame, mimetype='image.jpeg'), 200
    
@app_routes.route('/backend/sketch_boundary')
def drawing():
    """Pops up a cv window of the paused frame, click and hold left button to drop points as you drag,  x to exit"""
    global paused_cv_image, drawing_frame, paused_frame, points, paused_pc_msg
    # AttributeError: 'NoneType' object has no attribute 'copy'
    # Check if paused_cv_image is None
    logging.info("Sketch_boundary endpoint called (logging)")
    print("sketch boundary called (print)")

    if paused_cv_image is None:
        return "No paused image to draw on", 400
    
    try:
        drawing_frame = cv.UMat(paused_cv_image.copy())
    except AttributeError as e:
        print(f"Error copying paused_cv_image: {e}")
        return "Internal server error: paused_cv_image", 500
    
    #drawing_frame = cv.UMat(paused_cv_image.copy())

    cv.namedWindow("Video 1", cv.WINDOW_GUI_NORMAL | cv.WINDOW_AUTOSIZE)
    cv.moveWindow("Video 1", 500, 250)
    cv.setMouseCallback('Video 1', draw_polylines)
    
    logging.info("OpenCV window setup complete")   
    # Main loop
    is_drawing = True
    while is_drawing:
        #print('DRAWING')
        cv.imshow("Video 1", drawing_frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('x') or key == 27:
            #print('STOPPED')
            is_drawing = False
    logging.info("OpenCV window closed")
    # Clean up
    cv.destroyAllWindows()
    # Run the OpenCV window in a separate thread
    # window_thread = threading.Thread(target=display_window)
    # window_thread.start()
    # window_thread.join()
    # Project the sketch into the ego lidar
    try:
        projected_pcd = project_sketch(points, paused_pc_msg)
        #pcd = o3d.io.read_point_cloud("../../test_data/fragment.pcd")
        o3d.io.write_point_cloud("UI/projected_pcd.pcd", projected_pcd) 
    except Exception as e:
        print(f"Function project_sketch failed: {e}")
    # Reinitialize points
    points = []
    logging.info("Points reinitialized\n")
    return Response(paused_frame, mimetype='image.jpeg'), 200


@app_routes.route('/backend/process_message', methods=['POST'])
def process_message():
    data = request.get_json()
    print(data)
    print()

    return jsonify(data)

# @app_routes.route("/") 
# def test():
#     return "Hi"

###################### websocket ###################3
@socketio.on('incoming')
def handle_messsage(message):
    print("GPT message recieved through socket\n")
    logging.info('Received message (incoming):\n {message} \n')
    # Process message and send response if needed
    socketio.emit('incoming',message) 

@socketio.on('outgoing')
def handle_outgoing(message):
    print("User sent feedback through socket\n")
    logging.info(f'Received message (outgoing):\n {message} \n')
    # Process message and send response if needed
    socketio.emit('outgoing', message)  

# @socketio.on('sketch_proj_points')
    # socketio.emit('sketch_proj_points', sketch_proj)

