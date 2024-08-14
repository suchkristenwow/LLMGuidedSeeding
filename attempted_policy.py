from LLMGuidedSeeding_pkg.robot_client.simBot import simBot 
from LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms 
from LLMGuidedSeeding_pkg.utils.rehearsal_utils import * 
from chatGPT_written_utils import * 

from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points

# Instantiate the simBot and robotTransforms classes
bot = simBot(config_path, plot_bounds, init_pose, target_locations, obstacle_locations)
robot_transforms = robotTransforms(config_path)

# 1. Determine if the current position of the robot is within the plot bounds using lidar-inertial odometry and GPS localization.
if not bot.in_plot_bounds():
    # 3. Begin planning the shortest route to enter the bounded area if the robot is currently outside the plot bounds.
    nearest_point_on_perimeter, _ = nearest_points(Polygon(bot.plot_bounds), Point(bot.get_current_pose()[0], bot.get_current_pose()[1]))
    # Set waypoint 
    bot.current_waypoint = np.zeros((6,)) 
    bot.current_waypoint[0] = nearest_point_on_perimeter.x
    bot.current_waypoint[1] = nearest_point_on_perimeter.y
    heading = np.arctan2(nearest_point_on_perimeter.y - bot.get_current_pose()[1], nearest_point_on_perimeter.x - bot.get_current_pose()[0])  
    bot.current_waypoint[5] = heading 
    bot.go_to_waypoint() 

# 2. Check if there are any obstacles such as previously planted areas or conmods in the immediate vicinity that need to be avoided.
# Add your own implementation to avoid obstacles

# 4. Plant seeds in a 1mx1m grid pattern within the bounded area.
# Create a 1mx1m grid within the plot bounds
grid_x = np.arange(bot.plot_bounds[:, 0].min(), bot.plot_bounds[:, 0].max(), 1)
grid_y = np.arange(bot.plot_bounds[:, 1].min(), bot.plot_bounds[:, 1].max(), 1)
grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1,2)

# Iterate through all points in the grid
for point in grid_points:
    # Check if the point is within the plot bounds
    if Polygon(bot.plot_bounds).contains(Point(point)):
        # Plan the path to the point
        bot.current_waypoint[0] = point[0]
        bot.current_waypoint[1] = point[1]
        heading = np.arctan2(point[1] - bot.get_current_pose()[1], point[0] - bot.get_current_pose()[0])  
        bot.current_waypoint[5] = heading
        bot.go_to_waypoint()

        # Plant the seed
        bot.plant()

# Continuously monitor the robot's position and adjust its trajectory to maintain the 1mx1m grid pattern while planting seeds.
# Add your own implementation to monitor the robot's position and adjust its trajectory