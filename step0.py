import numpy as np 
from LLMGuidedSeeding_pkg.robot_client.simBot import simBot 
from LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms 
from LLMGuidedSeeding_pkg.utils.rehearsal_utils import * 
from chatGPT_written_utils import * 

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

# Instantiate the SimBot
config_path = "configs/example_config.toml"
plot_bounds_path = "random_path.csv"
target_locations =  {} 
obstacle_locations = {'conmods': [[11.43642048571068, 15.89478775336405], [-7.2985692415841115, -6.636902086955595], [0.8460524233551361, -2.3496021956321425], [-16.091427902492832, 18.480771921778143], [-13.70329818099941, 18.98390262888714], [1.28299135175466, -3.7833144505423686], [13.997710099529822, -9.08790856519363]]} 

plot_bounds = np.genfromtxt(plot_bounds_path,delimiter=",")  
plot_bounds = plot_bounds[~np.isnan(plot_bounds).any(axis=1)]
if not np.array_equal(plot_bounds[0], plot_bounds[-1]):
    plot_bounds = np.vstack([plot_bounds, plot_bounds[0]]) 

init_pose = [    -15.213,      21.119,           0 ,          0 ,          0   ,    2.207]
bot = simBot(config_path, plot_bounds, init_pose, target_locations, obstacle_locations)

# Determine if the robot is within the plot bounds
if not bot.in_plot_bounds():
    # The robot is outside the plot bounds
    # Find the shortest route to the plot bounds
    robot_pose = bot.get_current_pose()
    nearest_point_on_perimeter, _ = nearest_points(Polygon(bot.plot_bounds), Point(robot_pose[0],robot_pose[1]))
    bot.current_waypoint = np.array([nearest_point_on_perimeter.x, nearest_point_on_perimeter.y, 0, 0, 0, 0])
    bot.go_to_waypoint()

# Check for obstacles in the vicinity
observed_obstacles = []
if "obstacle" in bot.current_map.keys():
    observed_obstacles = bot.current_map["obstacle"]

# Begin the planting process
grid_size = 1  # 1m x 1m grid
for x in np.arange(plot_bounds[0, 0], plot_bounds[-1, 0], grid_size):
    for y in np.arange(plot_bounds[0, 1], plot_bounds[-1, 1], grid_size):
        waypoint = np.array([x, y, 0, 0, 0, 0])
        obstacle_nearby = False
        for obstacle in observed_obstacles:
            if np.linalg.norm(obstacle.mu - waypoint[:2]) < grid_size:
                obstacle_nearby = True
                break
        if not obstacle_nearby:
            # No obstacle nearby, plant seeds
            bot.current_waypoint = waypoint
            bot.go_to_waypoint()
            bot.plant()