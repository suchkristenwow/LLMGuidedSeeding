import math 
import numpy as np 
from shapely.geometry import Point, Polygon 
from shapely.ops import nearest_points 

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