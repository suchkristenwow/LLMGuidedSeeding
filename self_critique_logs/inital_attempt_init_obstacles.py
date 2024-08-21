from LLMGuidedSeeding_pkg.robot_client.simBot import simBot 
from LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms 
from LLMGuidedSeeding_pkg.utils.rehearsal_utils import * 
from chatGPT_written_utils import * 

from shapely.geometry import Polygon, Point
import numpy as np

# Standard width of tape in meters
tape_width = 0.0254  # 1 inch is 0.0254 meters
# Let's assume the tape length can vary between 0.1 and 1 meter
min_tape_length = 0.1
max_tape_length = 1.0

# We will use the gen_random_points_in_plot_bounds function to generate the center points of the tapes
tape_centers = gen_random_points_in_plot_bounds(plot_bounds, n_obstacles)

# Initialize the list of obstacles
obstacles = []

for center in tape_centers:
    # Generate a random length for the tape
    tape_length = np.random.uniform(min_tape_length, max_tape_length)
    
    # Define the four corners of the rectangle
    points = [
        (center[0] - tape_length / 2, center[1] - tape_width / 2),  # Bottom left
        (center[0] - tape_length / 2, center[1] + tape_width / 2),  # Top left
        (center[0] + tape_length / 2, center[1] + tape_width / 2),  # Top right
        (center[0] + tape_length / 2, center[1] - tape_width / 2),  # Bottom right
    ]
    
    # Create the Polygon object
    tape = Polygon(points)
    
    # Check for overlap with existing landmarks
    if not check_overlap_w_existing_lms(tape, existing_landmarks):
        # If there is no overlap, add the tape to the list of obstacles
        obstacles.append(tape)