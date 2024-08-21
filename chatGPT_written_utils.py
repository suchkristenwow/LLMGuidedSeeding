import math 
import numpy as np 
from shapely.geometry import Point, Polygon 
from shapely.ops import nearest_points 
from matplotlib.path import Path  

def generate_circle_points(center, radius, n=5):
    """
    Generate n points on the perimeter of a circle.

    :param radius: Radius of the circle
    :param center: Tuple representing the center of the circle (x, y)
    :param n: Number of points to generate
    :return: List of tuples representing the points on the circle perimeter
    """
    points = []
    angle_increment = 2 * math.pi / n

    for i in range(n):
        angle = i * angle_increment
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))

    return points

def plant_with_offset(tfs,pose,coord,offset):
    shrub_pts = generate_circle_points(coord,offset)   
    if not np.array_equal(shrub_pts[0],shrub_pts[-1]):
        shrub_pts = np.vstack([shrub_pts,shrub_pts[0]])
    shrub_perimeter = Polygon(shrub_pts) 
    nearest_shrub_pt, _ = nearest_points(shrub_perimeter, Point(pose[0],pose[1]))  
    heading = np.arctan2(nearest_shrub_pt.y - pose[1], nearest_shrub_pt.x - pose[0])  
    current_waypoint = tfs.get_robot_pose_from_planter(nearest_shrub_pt) 
    return current_waypoint 

def get_id_given_coord(objects,coord): 
    '''
    Want to determine the id of the coord we planted at 
    '''
    # Find the closest existing object using Euclidean distance
    distances = [np.linalg.norm(x.mu - coord) for x in objects] 
    idx = np.argmin(distances) 
    return objects[idx].object_id

def check_overlap_w_existing_lms(shape,existing_landmarks): 
    """
    Return true if there is overlap 
    """
    overlap_found = any(shape.intersects(landmark) for landmark in existing_landmarks)
    return overlap_found 

def gen_random_points_in_plot_bounds(plot_bounds,num_points): 
    if not isinstance(plot_bounds,np.ndarray): 
        coords = plot_bounds.exterior.coords 
        plot_bounds = np.array([[[x[0],x[1]]] for x in coords]); 
        plot_bounds = np.reshape(plot_bounds,(len(coords),2))
    
    if len(plot_bounds.shape) > 2:
        plot_bounds = np.reshape(plot_bounds,(len(plot_bounds,2))) 
        plot_bounds = np.squeeze(plot_bounds)  

    min_x = min(plot_bounds[:,0]); max_x = max(plot_bounds[:,0]); 
    min_y = min(plot_bounds[:,1]); max_y = max(plot_bounds[:,1])

    contour = Path(plot_bounds) 

    points = []
    while len(points) < num_points:
        # Generate random points within the bounding box
        random_points = np.random.rand(num_points, 2)
        random_points[:, 0] = random_points[:, 0] * (max_x - min_x) + min_x
        random_points[:, 1] = random_points[:, 1] * (max_y - min_y) + min_y
        
        # Check which points are inside the contour
        mask = contour.contains_points(random_points)
        points_inside = random_points[mask]
        
        # Add the valid points to the list
        points.extend(points_inside.tolist())
        
        # Limit the number of points to the desired number
        points = points[:num_points]
    return points 