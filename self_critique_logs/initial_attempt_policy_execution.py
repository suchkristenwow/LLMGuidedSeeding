import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull
from operator import itemgetter

def get_six_even_points(bounds):
    '''
    Function to get six evenly spaced points in the given plot bounds
    Args:
        bounds: plot bounds
    Returns:
        list of six evenly spaced points in the plot
    '''
    hull = ConvexHull(bounds)
    points = hull.points
    points = sorted(points, key=itemgetter(0, 1))
    return points[::len(points)//6]

# Initialize simBot
robot = simBot(config_path, plot_bounds, init_pose, targets, obstacles)

pose = robot.get_current_pose()

# Step 1: Initialize the task by checking the robot's current location within the plot bounds.
if not robot.in_plot_bounds():
    nearest_point_on_perimeter, _ = nearest_points(Polygon(robot.plot_bounds), Point(pose[0], pose[1])) 
    robot.current_waypoint = np.zeros((6,))
    robot.current_waypoint[0] = nearest_point_on_perimeter.x
    robot.current_waypoint[1] = nearest_point_on_perimeter.y
    heading = np.arctan2(nearest_point_on_perimeter.y - pose[1], nearest_point_on_perimeter.x - pose[0])
    robot.current_waypoint[5] = heading
    if not robot.go_to_waypoint():
        raise RuntimeError("Failed to move to the nearest point within plot bounds.")

# Step 2: Find 6 evenly spaced points within the plot bounds.
six_points = get_six_even_points(plot_bounds)
print("six_points: ", six_points)

# Step 3: Navigate the robot to the points and plant at those points, avoiding the blue tape.
while len(robot.planted_locations) < 6:
    for i, point in enumerate(six_points):
        # Get waypoint to the point
        waypoint = plant_with_offset(robot.static_transformer, robot.get_current_pose(), point, 0)
        
        # Check if the waypoint is inside the plot bounds
        if not Polygon(robot.plot_bounds).contains(Point(waypoint[0], waypoint[1])):
            print(f"Waypoint {waypoint} is outside the plot bounds, replanning.")
            continue  # Skip to the next iteration to replan

        robot.current_waypoint = waypoint
        
        # Attempt to move to the waypoint
        if robot.go_to_waypoint():
            # Plant at the waypoint if the robot successfully reaches it
            robot.plant()
            
            # Check if the required number of plants has been reached
            if len(robot.planted_locations) >= 6:
                break
        else:
            print(f"Failed to reach waypoint {waypoint}, replanning.")
