from LLMGuidedSeeding_pkg.robot_client.simBot import simBot 
from LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms 
from LLMGuidedSeeding_pkg.utils.rehearsal_utils import * 
from chatGPT_written_utils import * 

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
from scipy.spatial import ConvexHull

# Initialize simBot
robot = simBot(config_path, plot_bounds, init_pose, targets, obstacles)

# Step 1: Initialize the task by checking the robot's current location within the plot bounds.
pose = robot.get_current_pose()
if not robot.in_plot_bounds():
    nearest_point_on_perimeter = nearest_points(Polygon(np.array(plot_bounds)), Point(pose[0], pose[1]))[1]
    robot.current_waypoint = np.array([nearest_point_on_perimeter.x, nearest_point_on_perimeter.y, 0, 0, 0, 0])
    robot.go_to_waypoint()

# Function to get six evenly spaced points in the given plot bounds
def get_six_even_points(bounds):
    hull = ConvexHull(bounds)
    points = hull.points
    points = sorted(points, key=lambda k: [k[0], k[1]])
    return points[::len(points)//6]

# Step 2: Find 6 evenly spaced points within the plot bounds.
six_points = get_six_even_points(plot_bounds)

# Step 3: Navigate the robot to the points and plant at those points, avoiding the blue tape.
for point in six_points:
    # Get waypoint to the point
    waypoint = plant_with_offset(robot.static_transformer, robot.get_current_pose(), point, 0)
    robot.current_waypoint = waypoint
    robot.go_to_waypoint()
    
    # Plant at the waypoint
    robot.plant()