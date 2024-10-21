import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import LinearRing

# Initialize Robot and robotTransforms instances
robot = Robot(config_path, plot_bounds, init_pose, targets, obstacles)
robot_transforms = robotTransforms(config_path)

# Step 1: Ensure the robot is within the plot bounds
if not robot.in_plot_bounds():
    # Get current robot pose
    robot_pose = robot.get_current_pose()

    # Get closest point on plot boundary
    ring = LinearRing(plot_bounds)
    nearest_point = ring.interpolate(ring.project(Point(robot_pose[0], robot_pose[1])))
    
    # Set current waypoint to nearest point
    robot.current_waypoint = np.array([nearest_point.x, nearest_point.y, robot_pose[2], robot_pose[3], robot_pose[4], robot_pose[5]])

    # Execute movement
    robot.go_to_waypoint()

# Step 2: Initialize empty list to store planting locations
robot.planted_locations = []

# Steps 3 to 9: Implement lawnmower pattern and plant at six points
tape_label = "yellow and black striped tape"
num_points = 6
points_planted = 0
while points_planted < num_points:
    # Step 3: Identify yellow and black striped tape
    tape_id, tape_loc = robot.check_environment_for_something(tape_label)
    
    # Step 4: Choose evenly spaced points within plot bounds
    x_range = np.ptp(plot_bounds[:, 0])
    y_range = np.ptp(plot_bounds[:, 1])
    x_spacing = x_range / (num_points + 1)
    y_spacing = y_range / (num_points + 1)
    x_coords = np.linspace(np.min(plot_bounds[:, 0]) + x_spacing, np.max(plot_bounds[:, 0]) - x_spacing, num_points)
    y_coords = np.linspace(np.min(plot_bounds[:, 1]) + y_spacing, np.max(plot_bounds[:, 1]) - y_spacing, num_points)
    
    # Step 5: Follow lawnmower pattern and check for tape
    for x, y in zip(x_coords, y_coords):
        # Set current waypoint
        robot.current_waypoint = np.array([x, y, robot_pose[2], robot_pose[3], robot_pose[4], robot_pose[5]])
        
        # If tape detected, replan path around tape
        if tape_id != -1:
            # Determine new waypoint to avoid tape
            new_waypoint = np.array([tape_loc[0] + (2 * tape_loc[0] - x), tape_loc[1] + (2 * tape_loc[1] - y), robot_pose[2], robot_pose[3], robot_pose[4], robot_pose[5]])
            
            # Set current waypoint to new waypoint
            robot.current_waypoint = new_waypoint
        
        # Step 6: Follow lawnmower pattern to next waypoint
        robot.go_to_waypoint()
        
        # Step 7: Plant seed at chosen point
        robot.plant()
        
        # Step 8: Record planting location
        robot.planted_locations.append([x, y])
        
        # Increment points planted counter
        points_planted += 1
        
        # Step 9: Continue process until all six seeds have been planted
        if points_planted >= num_points:
            break