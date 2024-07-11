import rosbag
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge
from transform import CamProjector
import rospy
import trimesh 
import matplotlib.pyplot as plt

def get_single_pc_msg(bag_file, topic, capture_time, epsilon):
    bag = rosbag.Bag(bag_file, 'r')
    for _, msg, t in bag.read_messages(topics = [topic]):
        if abs(t - capture_time).to_sec() < epsilon:
            gen = pc2.read_points(msg, skip_nans = True, field_names = ("x", "y","z")) # iterator object
            break
    bag.close()
    print('Bag Secured...')
    return np.array([point for point in gen])

def get_start_time(bag_file, topic):
    bag = rosbag.Bag(bag_file)
    start_time = None
    for topic_name, _, t in bag.read_messages(topics=[topic]):
        # Check if the current message is on the specified topic
        if topic_name == topic:
            # Update the start time if it's None or if the current message's timestamp is earlier
            if start_time is None or t < start_time:
                start_time = t

    # Close the bag file
    bag.close()

    return start_time

def load_points(from_csv, bag_file = '', img_topic = '', video_speed = 30):
    if from_csv:
        points = np.array([np.loadtxt('points.csv', dtype='int', delimiter=',', usecols=(0, 1), unpack=False)])
        points = points.reshape(-1, points.shape[-1])
        return points, get_start_time(bag_file, img_topic)
    else:
        points, capture_time = sketch_boundary(bag_file,img_topic, video_speed)
        return points, capture_time 

def point_cloud_from_pc_arrays(point_arrays, colors):
    point_clouds = []
    for points, color in zip(point_arrays, colors):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(np.array([color] * len(points))) # [R, G, B]
        point_clouds.append(point_cloud)
    return point_clouds

def voxel_from_pc_arrays(point_arrays, colors, voxel_size):
    pcds = point_cloud_from_pc_arrays(point_arrays, colors)
    voxel_grids = []
    for pcd in pcds:
        voxel_grids.append(o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=voxel_size))
    return voxel_grids
    
def voxel_from_pcd(pcds, voxel_size):
    voxel_grids = []
    for pcd in pcds:
        voxel_grids.append(o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size= voxel_size))
    return voxel_grids

def octree_from_pc_arrays(point_arrays, colors, size_expand):
    pcds = point_cloud_from_pc_arrays(point_arrays, colors)
    octrees = []
    for pcd in pcds:
        octree = o3d.geometry.Octree(max_depth = 6)
        octree.convert_from_point_cloud(pcd,size_expand)
        octrees.append(octree)
    return octrees
    
def mesh_from_pc_arrays(point_arrays, colors):
    pcds = point_cloud_from_pc_arrays(point_arrays, colors)
    mesh = []
    for pcd in pcds:
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
         # a high alpha so that we can have big triangles
        trimesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha = 1)
        # mesh = o3d.geometry.TriangleMesh(points)
        trimesh.compute_vertex_normals()    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii = o3d.utility.DoubleVector([0.5, 0.5]))
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = 8)
        #mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([color for _ in range(len(mesh.vertices))]))
        mesh.append(trimesh)
    return mesh

def mesh_from_pcd(pcds):
    mesh = []
    for pcd in pcds:
        print(type(pcd))
        trimesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha =  1)
        trimesh.compute_vertex_normals()
        mesh.append(trimesh)
    return mesh

def mesh_lineset_intersection(mesh, line_set):
    trimesh_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
    
    # Perform ray-mesh intersection, what if it doesn't intersect?
    locations, index_ray, index_tri = trimesh_mesh.ray.intersects_location(ray_origins = np.array(line_set.points[:len(line_set.points)//2]), 
                                                                            ray_directions=np.array(line_set.points[len(line_set.points)//2:]))
    return locations

def get_depth(floor_array, sketch_points):
    cam_model = CamProjector.get_camera_model()
    rays = []
    for pixel in sketch_points:
        rays.append(cam_model.projectPixelTo3dRay(pixel))
    return rays
    
def create_lines(start_points, end_points, color):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack([start_points, end_points]))
    lines_indices = np.column_stack([np.arange(len(start_points)), np.arange(len(start_points), len(start_points)*2)])
    line_set.lines = o3d.utility.Vector2iVector(lines_indices)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(line_set.lines))])
    return [line_set]

def select_pc_pixel_coordinates(cloud_array):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(cloud_array)
    print("-----------------------------------")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    picked_points = vis.get_picked_points()
    return np.array([cloud_array[index] for index in picked_points])

def sketch_boundary(bag_file, img_topic, video_speed = 30):
    # init some items for drawing purposes
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
    # open the bag file
    bag = rosbag.Bag(bag_file)
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
            cv.imshow("Video 1", cv_image1)
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
            cv.imshow("Video 1", cv_image1)
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

    
if __name__ == "__main__":
    bag_file = '/home/miles/Documents/bags/exp1.bag' # '/home/miles/Documents/bags/front_cam_g_truth.bag'
    img_topic =  '/H03/cam_front/image_color' #'/H03/horiz/os_cloud_node/points' # '/H03/cam_front/image_color' # cam_front/image_color'
    pc_topic = '/H03/horiz/os_cloud_node/points'

    # homogenous transformation matrices
    camera_tf = [0.152758, 0.00, 0.0324, 0,0,0] # [0.152758, 0.00, 0.0324, 0,0,0]
    robot_pose = [0,0,0,0,0,0] # np.pi/2
    
    # Draw on an image and get the pixel coordinates
    sketch_points, capture_time = load_points(from_csv = False, bag_file = bag_file, img_topic=img_topic, video_speed = 40)
    msg_points = get_single_pc_msg(bag_file, pc_topic, capture_time, epsilon = .1)
    
    # Origin point in point cloud
    origin_pcd = o3d.geometry.PointCloud()
    origin_pcd.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))
    
    # project sketched pixel coordinates into 3D from the camera parameters
    cam_projector = CamProjector(1, camera_pose=camera_tf, robot_pose=robot_pose)
    sketch_proj = np.array([cam_projector.project(c) for c in sketch_points])[:,:3]

    # Get transformed point of camera 
    camera_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(camera_tf))
    camera_pcd = o3d.geometry.PointCloud()
    camera_point = [camera_tf_mat[0,3], camera_tf_mat[1,3], camera_tf_mat[2,3]]
    camera_pcd.points = o3d.utility.Vector3dVector(np.array([camera_point]))
    
    # point cloud of the environment
    world_pcd = o3d.geometry.PointCloud()
    world_pcd.points = o3d.utility.Vector3dVector(msg_points)
    world_pcd.colors = o3d.utility.Vector3dVector([[.5,.5,.5] for _ in range(len(world_pcd.points))])
    
    # Project world points to the unit sphere
    world_distances = np.array(world_pcd.compute_point_cloud_distance(origin_pcd)) + .00001
    world_unit = np.asarray(world_pcd.points) / world_distances[:,None] 
 
    # world points on the unit sphere
    world_unit_pcd = o3d.geometry.PointCloud()
    world_unit_pcd.points = o3d.utility.Vector3dVector(world_unit)
    world_unit_pcd.colors = o3d.utility.Vector3dVector([[0,.5,.5] for _ in range(len(world_unit_pcd.points))])
    
    # sketched points projected into the point cloud space
    sketch_pcd = o3d.geometry.PointCloud()
    sketch_pcd.points = o3d.utility.Vector3dVector(sketch_proj)
    sketch_pcd.colors = o3d.utility.Vector3dVector(np.array([[1,0,0] for _ in range(len(sketch_pcd.points))]))
    
    # Project sketc points ot the unit sphere
    sketch_distances = np.array(sketch_pcd.compute_point_cloud_distance(origin_pcd)) + .0001
    sketch_unit = np.asarray(sketch_pcd.points) / sketch_distances[:,None]

    # sketch points on the unit sphere
    sketch_unit_pcd = o3d.geometry.PointCloud()
    sketch_unit_pcd.points = o3d.utility.Vector3dVector(sketch_unit)
    sketch_unit_pcd.colors = o3d.utility.Vector3dVector([1,0,.8] for _ in range(len(sketch_unit_pcd.points)))
    
    # creating the rays
    rays = create_lines(np.array([camera_point for _ in range(len(sketch_unit))]), sketch_unit, [.6,1,.6])

    # calculate sketch_unit distances from the world_unit points
    pcd_distances = np.sqrt(np.sum((world_unit - sketch_unit[:,np.newaxis,:])  ** 2, axis=2))
    closest_ind = np.argmin(pcd_distances, axis = 1)
    
    # print(pcd_distances, closest_ind)
    # print(np.array(pcd.points)[closest_ind])
    sketch_3d = o3d.geometry.PointCloud()
    sketch_3d.points = o3d.utility.Vector3dVector(np.array(world_pcd.points)[closest_ind])
    sketch_3d.colors = o3d.utility.Vector3dVector([[1,.5,0] for _ in range(len(sketch_3d.points))])

    world = np.delete(np.array(world_pcd.points), closest_ind, axis = 0)
    world_pcd.points = o3d.utility.Vector3dVector(world)
    world_pcd.colors = o3d.utility.Vector3dVector([[0,0,0] for _ in range(len(world_pcd.points))])
    #print(np.array(sketch_3d.points))
    
    # o3d.visualization.draw_geometries([world_pcd, world_unit_pcd, sketch_3d,
    #                                   sketch_unit_pcd, sketch_pcd, camera_pcd] + rays) # 
    #o3d.visualization.draw_geometries(octree)
    voxels = voxel_from_pcd([sketch_3d, world_pcd], .05)
    o3d.visualization.draw_geometries(voxels)

    octree = octree_from_pc_arrays([msg_points], [[0,0,0]], .05)
    #mesh = mesh_from_pcd([world])
    #mesh = mesh_from_pc_arrays([world], [[1,.5,0]])

    #o3d.visualization.draw_geometries(mesh)