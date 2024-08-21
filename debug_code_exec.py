from LLMGuidedSeeding_pkg.robot_client.simBot import simBot 
from LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms 
from LLMGuidedSeeding_pkg.utils.rehearsal_utils import * 
from chatGPT_written_utils import * 
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points

# Initialize the robot and transformations
robot = simBot(config_path, plot_bounds, init_pose, targets, obstacles)
tfs = robotTransforms(config_path)

# Define the plot as a polygon
plot_polygon = Polygon(plot_bounds)

# 1. Check robot's location within plot bounds
if not robot.in_plot_bounds():
    current_pose = robot.get_current_pose()
    nearest_point_on_perimeter, _ = nearest_points(plot_polygon, Point(current_pose[0:2]))
    robot.current_waypoint = np.concatenate([np.array(nearest_point_on_perimeter), [0, 0, 0, 0]], axis=None)
    robot.go_to_waypoint()

# 2. Find 6 evenly spaced points within plot bounds
line = LineString(plot_polygon.exterior.coords)

points = [line.interpolate(i/6, normalized=True) for i in range(6)]

# 3. Plant at those points, avoiding blue tape
for point in points:
    coord = np.array([point.x, point.y])
    waypoint = plant_with_offset(tfs,robot.get_current_pose(), coord, 0)
    robot.current_waypoint = waypoint
    robot.go_to_waypoint()
    robot.plant()