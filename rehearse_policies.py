from LLMGuidedSeeding_pkg import * 
import os 
import toml 
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import random 

def get_unique_markers(num_markers):
    """
    Get a list of unique random markers.
    
    Parameters:
    - num_markers: The number of unique markers to select.
    
    Returns:
    - A list of unique marker styles.
    """
    marker_styles = [
        '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
        'h', 'H', '+', 'x', 'D', 'd', '|', '_'
    ]
    
    if num_markers > len(marker_styles):
        raise ValueError("Number of requested markers exceeds the number of available unique markers.")
    
    return random.sample(marker_styles, num_markers)

def point_in_bounds(point,polygon): 
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside 

class policyRehearsal: 
    def __init__(self,args):
        self.prompt_path = args.prompt_path  
        self.logging_dir = args.logging_dir 
        self.plot_bounds_path = args.plot_bounds_path 
        self.parameters = toml.load(args.config_path) 
        plt.ion() 
        fig, ax = plt.subplots(figsize=(12,12)) #this is for the animation thing 
        self.fig = fig; self.ax = ax 
        self.robot_traj = np.zeros((1,3)) 
        self.obstacle_locations = {} 
        self.target_locations = {} 
        self.meta_constraints = {} 

    def plot_frame(self): 
        self.ax.clear() 
        # Plot Robot & Pointer, and Bounds 
        plot_bounds = np.genfromtxt(self.plot_bounds_path) 
        self.ax.plot(plot_bounds[:,0],plot_bounds[:,1],color="k") 
        maxX = max([max(self.robot_traj[:,0]),max(self.obstacle_locations[:,0]),max(self.target_locations[:,0])]) 
        minX = min([min(self.robot_traj[:,0]),min(self.obstacle_locations[:,0]),min(self.target_locations[:,0])]) 
        deltaX = maxX - minX 
        maxY = max([max(self.robot_traj[:,1]),max(self.obstacle_locations[:,1]),max(self.target_locations[:,1])]) 
        minY = min([min(self.robot_traj[:,1]),min(self.obstacle_locations[:,1]),min(self.target_locations[:,1])]) 
        deltaY = maxY - minY 
        bounds_mag = max([deltaX,deltaY]) 
        minX -= bounds_mag*0.1; maxX += bounds_mag*0.1 
        minY -= bounds_mag*0.1; maxY += bounds_mag*0.1 
        self.ax.set_xlim(minX,maxX)
        self.ax.set_ylim(minY,maxY) 
        self.ax.set_aspect('equal') 
        self.ax.scatter(self.robot_traj[-1,0],self.robot_traj[-1,0],color="k") 
        pointer_x = self.robot_traj[-1,0] + 2*np.cos(self.robot_traj[-1,2]); pointer_y = self.robot_traj[-1,1] + 2*np.sin(self.robot_traj[-1,2])  
        robot_pointer = self.ax.arrow(self.robot_traj[-1,0], self.robot_traj[-1,1], pointer_x, pointer_y, head_width=0.5, head_length=0.5, fc='red', ec='red') 
        self.ax.add_patch(robot_pointer) 
        # Plot traversed trajectory 
        self.ax.plot(self.robot_traj[:,0],self.robot_traj[:,1],linestyle="--")  
        # Plot any obstacles 
        obstacles = self.policy_constraints["avoid"] 
        obstacle_markers = get_unique_markers(len(obstacles)) 
        for i,obstacle_type in enumerate(obstacles): 
            for j,pt in enumerate(self.obstacle_locations[obstacle_type]): 
                if j == 0:
                    self.ax.scatter(pt[0],pt[1],color="red",marker=obstacle_markers[i],label=obstacle_type) 
                else: 
                    self.ax.scatter(pt[0],pt[1],color="red",marker=obstacle_markers[i]) 
        # Plot any targets 
        targets = self.policy_constraints["goal_lms"] 
        target_markers = get_unique_markers(len(obstacles)) 
        for i,lm_type in enumerate(targets): 
            for j,pt in enumerate(self.target_locations[lm_type]): 
                if j == 0:
                    self.ax.scatter(pt[0],pt[1],color="green",marker=target_markers[i],label=lm_type) 
                else: 
                    self.ax.scatter(pt[0],pt[1],color="green",marker=obstacle_markers[i]) 
        # Plot meta constraints 
        for constraint in self.meta_constraints.keys(): 
            if constraint["type"] == "point": 
                for pt in enumerate(constraint["locations"]):
                    self.ax.scatter(pt[0],pt[1],"purple",label=constraint) 
            else: 
                wheel_patches = get_robot_wheel_patches(self.robot_traj[-1,0],self.robot_traj[-1,1],self.robot_traj[-1,2],self.parameters["vehicle_params"])
                for patch in wheel_patches:
                    self.ax.add_patch(patch) 

    def plot_camera_views(self): 

    def rehearse(self): 
        #load in the polciy 
        with open(os.path.join(self.logging_directory,"finalPolicy.txt"),"r") as f:
            policy = f.read() 

        epoch = 0 
        while epoch < self.parameters["policyRehearsal"]["rehearsal_epochs"]: 

            epoch += 1 
       
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore Test")

    # Add an argument for test_query
    parser.add_argument(
        "--prompt_path",
        type=str,
        help="Path to desired prompt"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration file",
        default="experiment_runner/config/example_config.toml",
    )
    
    parser.add_argument(
        "--logging_dir",
        type=str,
        help="Path to the logging directory",
        default="logs",
    )

    parser.add_argument(
        "--plot_bounds_path",
        type=str,
        help="Path of a csv file containing the perimeter of the desired area of operation",
        default="random_path.csv",
    ) 

    # Parse the command-line arguments
    args = parser.parse_args()  

    pR = policyRehearsal(args)
    pR.rehearse()