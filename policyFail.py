from LLMGuidedSeeding_pkg.robot_client.simBot import simBot 
from LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms 
from LLMGuidedSeeding_pkg.utils.rehearsal_utils import * 
from chatGPT_written_utils import * 

import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import nearest_points
from scipy.spatial import ConvexHull

# Instantiate robot
bot = simBot(config_path, plot_bounds, init_pose, target_locations, obstacle_locations)

# Define function to plan the grid
def plan_grid(bounds, grid_size=1):
    min_x = np.min(bounds[:, 0])
    max_x = np.max(bounds[:, 0])
    min_y = np.min(bounds[:, 1])
    max_y = np.max(bounds[:, 1])

    x_coords = np.arange(min_x, max_x, grid_size)
    y_coords = np.arange(min_y, max_y, grid_size)

    return [(x, y) for x in x_coords for y in y_coords]

# Plan the grid
grid_points = plan_grid(plot_bounds)

# Loop through each point in the grid
for point in grid_points:
    # Check if robot is within bounds, if not, plan route to nearest point within bounds
    if not bot.in_plot_bounds():
        nearest_point_on_perimeter, _ = nearest_points(Polygon(bot.plot_bounds), Point(*point))
        bot.current_waypoint = np.array([nearest_point_on_perimeter.x, nearest_point_on_perimeter.y, 0, 0, 0, 0])
        bot.go_to_waypoint()

    # Check for obstacles and plan path around them
    if bot.check_environment_for_something('obstacle') is not None:
        _, obstacle_location = bot.check_environment_for_something('obstacle')
        bot.current_waypoint = bot.plan_path_around_obstacle(obstacle_location)
        bot.go_to_waypoint()

    # Plant seeds
    bot.plant()

    # Check for new obstacles in the immediate vicinity and replan path if necessary
    if bot.check_environment_for_something('obstacle') is not None:
        _, obstacle_location = bot.check_environment_for_something('obstacle')
        bot.current_waypoint = bot.plan_path_around_obstacle(obstacle_location)
        bot.go_to_waypoint()