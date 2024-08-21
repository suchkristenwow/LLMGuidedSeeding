

with open("prompts/salvage_prompt.txt","r") as f: 
    critique_prompt =f.read()

insert_dict = {'*INSERT_DESCRIPTION*': 'The user has provided a description of the following objects which are important to completing this task: \nTape comes in rectangular strips. Blue tape is blue. The tape is on the floor. There could be large patches of it, or long skinny strips. The width is about 1 in and the \nlength is variable. \n', '*INSERT_OBJECT*': "['blue tape']", '*INSERT_IMPORTS*': 'from LLMGuidedSeeding_pkg.robot_client.simBot import simBot \nfrom LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms \nfrom LLMGuidedSeeding_pkg.utils.rehearsal_utils import * \nfrom chatGPT_written_utils import * \n', '*INSERT_CODE*': "from shapely.geometry import Point, Polygon\nfrom shapely.ops import nearest_points\nimport numpy as np\n\n# Instantiate the simBot and robotTransforms classes\nbot = simBot(config_path, plot_bounds, init_pose, targets, obstacles)\nrobot_transforms = robotTransforms(config_path)\n\n# Step 1: Check if the robot's current location is within the plot bounds. \n# If it is not, plan the shortest route to enter the plot bounds.\ncurrent_pose = bot.get_current_pose()\nif not bot.in_plot_bounds():\n    nearest_point_on_boundary, _ = nearest_points(Polygon(plot_bounds), Point(current_pose[0], current_pose[1]))\n    bot.current_waypoint = np.array([nearest_point_on_boundary.x, nearest_point_on_boundary.y, 0, 0, 0, 0])\n    bot.go_to_waypoint()\n\n# Step 2: Find 6 evenly spaced points within the plot bounds.\npoints_on_circle = generate_circle_points((0, 0), 1, 6)\nevenly_spaced_points = []\nfor point in points_on_circle:\n    nearest_point_on_boundary, _ = nearest_points(Polygon(plot_bounds), Point(point[0], point[1]))\n    evenly_spaced_points.append([nearest_point_on_boundary.x, nearest_point_on_boundary.y])\n\n# Step 3: Navigate the robot to plant at those points, avoiding driving over the blue tape.\nfor point in evenly_spaced_points:\n    waypoint = np.array([point[0], point[1], 0, 0, 0, 0])\n    bot.current_waypoint = waypoint\n    bot.go_to_waypoint()\n    bot.plant()", '*INSERT_ERROR*': "'dict' object has no attribute 'contains'"}

for thing in insert_dict:
    print(critique_prompt.find(thing)) 
    print("replacing {}".format(thing)) 
    critique_prompt.replace(thing,insert_dict[thing]) 
    

print("critique_prompt: ",critique_prompt)
