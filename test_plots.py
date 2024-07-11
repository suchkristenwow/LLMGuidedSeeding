import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def plot_robot_wheels(x, y, heading, wheel_base=0.75, wheel_track=1.0, wheel_width=0.2, wheel_length=0.4):
    """
    Plot the wheels of a robot given its position and heading.
    
    Parameters:
    - x: The x-coordinate of the robot.
    - y: The y-coordinate of the robot.
    - heading: The heading of the robot in radians.
    - wheel_base: The distance between the front and rear wheels.
    - wheel_track: The distance between the left and right wheels.
    - wheel_width: The width of each wheel.
    - wheel_length: The length of each wheel.
    """
    # Calculate the positions of the wheels
    half_wheel_base = wheel_base / 2.0
    half_wheel_track = wheel_track / 2.0
    
    # Front wheels
    front_left_wheel_x = x + half_wheel_base * np.cos(heading) - half_wheel_track * np.sin(heading)
    front_left_wheel_y = y + half_wheel_base * np.sin(heading) + half_wheel_track * np.cos(heading)
    
    front_right_wheel_x = x + half_wheel_base * np.cos(heading) + half_wheel_track * np.sin(heading)
    front_right_wheel_y = y + half_wheel_base * np.sin(heading) - half_wheel_track * np.cos(heading)
    
    # Rear wheels
    rear_left_wheel_x = x - half_wheel_base * np.cos(heading) - half_wheel_track * np.sin(heading)
    rear_left_wheel_y = y - half_wheel_base * np.sin(heading) + half_wheel_track * np.cos(heading)
    
    rear_right_wheel_x = x - half_wheel_base * np.cos(heading) + half_wheel_track * np.sin(heading)
    rear_right_wheel_y = y - half_wheel_base * np.sin(heading) - half_wheel_track * np.cos(heading)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create rectangles for each wheel
    wheels = [
        patches.Rectangle((front_left_wheel_x - wheel_length/2, front_left_wheel_y - wheel_width/2), wheel_length, wheel_width, 
                          angle=np.degrees(heading), linewidth=1, edgecolor='None', facecolor='blue', alpha=0.25, label='Front Left Wheel'),
        patches.Rectangle((front_right_wheel_x - wheel_length/2, front_right_wheel_y - wheel_width/2), wheel_length, wheel_width, 
                          angle=np.degrees(heading), linewidth=1, edgecolor='None', facecolor='blue',  alpha=0.25, label='Front Right Wheel'),
        patches.Rectangle((rear_left_wheel_x - wheel_length/2, rear_left_wheel_y - wheel_width/2), wheel_length, wheel_width, 
                          angle=np.degrees(heading), linewidth=1, edgecolor='None', facecolor='blue',  alpha=0.25, label='Rear Left Wheel'),
        patches.Rectangle((rear_right_wheel_x - wheel_length/2, rear_right_wheel_y - wheel_width/2), wheel_length, wheel_width, 
                          angle=np.degrees(heading), linewidth=1, edgecolor='None', facecolor='blue',  alpha=0.25, label='Rear Right Wheel')
    ]
    
    # Add rectangles to the plot
    for wheel in wheels:
        ax.add_patch(wheel)
    
    # Plot the robot position
    ax.plot(x, y, 'go', markersize=10, label='Robot Position')
    
    # Set plot limits and labels
    ax.set_xlim(min(front_left_wheel_x, front_right_wheel_x, rear_left_wheel_x, rear_right_wheel_x) - 1, 
                max(front_left_wheel_x, front_right_wheel_x, rear_left_wheel_x, rear_right_wheel_x) + 1)
    ax.set_ylim(min(front_left_wheel_y, front_right_wheel_y, rear_left_wheel_y, rear_right_wheel_y) - 1, 
                max(front_left_wheel_y, front_right_wheel_y, rear_left_wheel_y, rear_right_wheel_y) + 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_title('Robot Wheels')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    # Show the plot
    plt.show(block=True)

# Example usage:
x = 5
y = 5
heading = np.pi / 4  # 45 degrees in radians

plot_robot_wheels(x, y, heading)