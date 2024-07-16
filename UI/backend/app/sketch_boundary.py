import cv2 as cv
import rosbag
from cv_bridge import CvBridge
import numpy as np

def sketch_boundary(cam_model, bag_file, img_topic, video_speed = 30):
    # init some global vars for drawing purposes
    global drawing, points
    drawing = False
    points = []
    capture_time = 0
    # logic for our callback: Hold down the left click to drop points as you move the mouse
    def draw_polylines(event, x, y, flags, param):
        global ix, iy, drawing, points # we have to run the global call again in this function
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing == True:
                # we need to gather points from the user 
                cv.circle(cv_image1,(x,y),1,(0,0,255),-1)
                points.append([x,y])
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
        # if we want the dots to persist across the frames then we need to move cv.polylines to be inside opening messages loop
        elif event == cv.EVENT_RBUTTONDOWN:  
            points = np.array(points, np.int32)
            points = points.reshape((-1, 1, 2))
            cv.polylines(cv_image1, [points], isClosed=True, color=(0, 255, 0), thickness=3)
        # drop a single point
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(cv_image1, (x,y), 3, (0,255,0), -1)
            points.append([x,y])
# init the bridge between the bag file and OpenCV
    bridge = CvBridge()
    # Create a window to display the video
    cv.namedWindow("Video 1", cv.WINDOW_NORMAL)
    # Set the callback function that allows us to draw on the image
    cv.setMouseCallback('Video 1',draw_polylines,capture_time)
    
    
    # step through each message and visualize in the same window
    paused = False
    for _, msg, t in bag.read_messages(topics = [img_topic]):
        if not paused: 
            cv_image1 = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            #cv_image2 = bridge.imgmsg_to_cv2(msg2, desired_encoding='passthrough')
            rectified_image = cam_model.rectifyImage(cv_image1)
            cv.imshow("Video 1", rectified_image)
            #cv2.imshow("Video 2", cv_image2)
            # the speed at which we wait determines how quickly the video goes. If we can align this with how often this topic collected data we can scale to realtime
            key = cv.waitKey(video_speed)
            if key == ord(' '):
                capture_time = t
                paused = not paused
                if paused:
                    print("Paused")
                else:
                    print("Running")    
        #print(key)
        while paused:
            cv_image1 = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            rectified_image = cam_model.rectifyImage(cv_image1)
            cv.imshow("Video 1", rectified_image)
            key = cv.waitKey(video_speed)
            if key == ord(' '):
                paused = not paused
                if paused:
                    print("Paused")
                else:
                    print("Running")
            elif key == 27:  # 27 is the esc key, 20 is the "\n" key which is the enter key, 
                break
        if key == 27:
            break
    bag.close()
    cv.destroyAllWindows()
    points = np.array(points)
    points = points.reshape(-1, points.shape[-1])
    np.savetxt('points.csv', points, delimiter=',', fmt='%d')
    print(f'Points Saved at rospy time: {capture_time}')
    return points, capture_time