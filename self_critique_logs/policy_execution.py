from LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

# Initialize the robot and robotTransforms instances
bot = Robot(config_path, plot_bounds, init_pose, targets, obstacles)
robot_transforms = robotTransforms(config_path)

# 1. Check the robot's current location and move inside the plot if not already inside
if not bot.in_plot_bounds():
    current_pose = bot.get_current_pose()
    nearest_point_on_perimeter, _ = nearest_points(Polygon(bot.plot_bounds), Point(current_pose[0],current_pose[1]))
    bot.current_waypoint = np.array([nearest_point_on_perimeter.x, nearest_point_on_perimeter.y, 0, 0, 0, 0])
    bot.go_to_waypoint()

# Define pattern and pattern_offset
pattern = 'grid'
pattern_offset = 1

# Initialize a list to store the planted areas
bot.planted_locations = []

# Loop until the entire plot is covered in the grid pattern
while not bot.check_all_observed():
    # 2. Check for the presence of 'conmods' or previously planted areas
    bot.get_current_observations()
    
    # 3. Avoid 'conmods' or planted areas while moving towards the next planting location
    for label in bot.current_map:
        if label == 'conmods' or label == 'planted area':
            bot.current_waypoint = bot.get_current_pose()  # Stay in place if 'conmods' or planted areas are detected
            bot.go_to_waypoint()
    
    # 4. Move in a grid pattern with 1 meter between each planting spot
    if pattern == 'grid':
        current_pose = bot.get_current_pose()
        next_pose = current_pose
        next_pose[0] += pattern_offset  # Move in x direction by pattern_offset
        bot.current_waypoint = next_pose
        bot.go_to_waypoint()
    
    # 5. Plant at the next spot if the 'seed' value is True
    if 'seed' in constraints and constraints['seed'] == True:
        bot.plant()
        # 6. Store the current location as a 'planted area'
        bot.planted_locations.append(bot.get_current_pose()[:2])  # Store only x and y coordinates

    # 7. Repeat steps 2 to 6 until the entire plot within the bounds is covered in the grid pattern

# 9. Report the areas that have been successfully planted
print("Successfully planted in the following areas:")
for location in bot.planted_locations:
    print(f"x: {location[0]}, y: {location[1]}")