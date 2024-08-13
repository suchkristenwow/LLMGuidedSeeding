from LLMGuidedSeeding_pkg import * 
from LLMGuidedSeeding_pkg.utils import * 
from UI import ConversationalInterface 
import os 
import toml 
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import random 
from matplotlib.path import Path 
import pandas as pd 
import shutil 

class policyRehearsal: 
    def __init__(self,args):
        self.final_policy_path = args.final_policy_path
        self.prompt_path = args.prompt_path  
        self.logging_dir = args.logging_dir  
        self.plot_bounds = np.genfromtxt(args.plot_bounds_path,delimiter=",")  
        self.plot_bounds = self.plot_bounds[~np.isnan(self.plot_bounds).any(axis=1)]

        if not np.array_equal(self.plot_bounds[0], self.plot_bounds[-1]):
            self.plot_bounds = np.vstack([self.plot_bounds, self.plot_bounds[0]]) 

        self.parameters = toml.load(args.config_path) 
        self.maxD = self.parameters["robot"]["frustrum_length"] 
        #self.fov = self.parameters["robot"]["front_camera_fov_deg"]
        self.robot_length = self.parameters["robot"]["husky_length"]
        self.robot_width = self.parameters["robot"]["husky_width"]
        self.robot_traj = np.zeros((1,6)) 
        self.obstacle_locations = {} 
        self.target_locations = {} 
        self.meta_constraints = {} 
        
        #PLOTTING 
        plt.ion() 
        fig, ax = plt.subplots(figsize=(12,12)) #this is for the BEV animation thing 
        self.fig = fig; self.ax = ax 

        #UI ... idk how to integrate this 
        #self.conversational_interface = ConversationalInterface() 

        #SIM 
        #config_path,plot_bounds,target_locations,obstacle_locations
        self.simBot = simBot(args.config_path,self.plot_bounds,self.target_locations,self.obstacle_locations)

    def plot_robot(self):
        self.ax.scatter(self.robot_traj[-1,0],self.robot_traj[-1,1],color="k")  
    
        x = self.robot_traj[-1,0]; y = self.robot_traj[-1,1]; yaw = self.robot_traj[-1,2] 
            
        # Calculate the bottom-left corner of the rectangle considering the yaw angle
        corner_x = x - self.robot_length * np.cos(yaw) + (self.robot_width / 2) * np.sin(yaw)
        corner_y = y - self.robot_length * np.sin(yaw) - (self.robot_width / 2) * np.cos(yaw) 

        # Create the rectangle patch
        robot_rect = patches.Rectangle(
            (corner_x, corner_y), self.robot_length, self.robot_width,
            angle=np.degrees(yaw), edgecolor='black', facecolor='yellow', alpha=0.5
        )

        # Add the rectangle to the plot
        self.ax.add_patch(robot_rect)

        # Add an arrow to indicate the heading
        arrow_length = 0.5 * self.robot_length
        arrow_dx = arrow_length * np.cos(yaw)
        arrow_dy = arrow_length * np.sin(yaw)
        self.ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.1, head_length=0.1, fc='k', ec='k')         

    def plot_frame(self,tstep): 
        #Update trajectory 
        self.robot_traj = self.simBot.traj_cache 

        self.ax.clear() 
        # Plot Robot & Pointer, and Bounds 
        self.ax.plot(self.plot_bounds[:,0],self.plot_bounds[:,1],color="k") 

        self.ax.set_aspect('equal') 
        
        self.plot_robot()
        
        # Plot traversed trajectory 
        self.ax.plot(self.robot_traj[:,0],self.robot_traj[:,1],linestyle="--")  

        # Plot the constraints & obstacles 
        for obstacle in self.obstacle_locations: 
            X = self.obstacle_locations[obstacle]
            for i,x in enumerate(X): 
                if i == 0:
                    self.ax.scatter(x[0],x[1],color="red",marker="*",label=obstacle)
                else: 
                    self.ax.scatter(x[0],x[1],color="red",marker="*")

        for target in self.target_locations: 
            X = self.target_locations[target]
            for i,x in enumerate(X): 
                if i==0:
                    self.ax.scatter(x[0],x[1],color="green",label=target)  
                else: 
                    self.ax.scatter(x[0],x[1],color="green")

        #Plot the observations 
        if tstep in self.simBot.observation_cache: 
            observations_t = self.simBot.observation_cache[tstep]  
            for i in range(len(observations_t["front"]["coords"])): 
                observed_coord = observations_t["front"]["coords"][i]
                self.ax.plot([self.robot_traj[-1,0], observed_coord[0]],[self.robot_traj[-1,1], observed_coord[1]],color="red",linestyle="--")
            for i in range(len(observations_t["left"]["coords"])): 
                observed_coord = observations_t["left"]["coords"][i]
                self.ax.plot([self.robot_traj[-1,0], observed_coord[0]],[self.robot_traj[-1,1], observed_coord[1]],color="red",linestyle="--")
            for i in range(len(observations_t["right"]["coords"])): 
                observed_coord = observations_t["right"]["coords"][i]
                self.ax.plot([self.robot_traj[-1,0], observed_coord[0]],[self.robot_traj[-1,1], observed_coord[1]],color="red",linestyle="--")
        
        plt.legend() 
        plt.pause(0.05) 

        #self.fig.savefig("test_frames/frame"+str(tstep).zfill(5)+".png") 

    def plot_camera_frustrum(self,camera_type): 
        robot_pose = self.robot_traj[-1,:]
        fov = self.tfs.get_front_cam_fov(camera_type,robot_pose)
        p0 = fov[0]; p1 = fov[1]; p2 = fov[2]; p3 = fov[3]
        x0 = p0[0]; y0 = p0[1]; x1 = p1[0]; y1 = p1[1]; x2 = p2[0]; y2 = p2[1]; x3 = p3[0]; y3 = p3[1] 
        self.ax.plot([x0,x1],[y0,y1],'b')
        self.ax.plot([x1,x2],[y1,y2],'b')
        self.ax.plot([x2,x3],[y2,y3],'b')
        self.ax.plot([x3,x0],[y3,y0],'b')   

    def init_obstacles(self,obstacles):  
        contour = Path(self.plot_bounds)
        # Get the bounding box of the contour
        min_x, min_y = np.min(self.plot_bounds, axis=0)
        max_x, max_y = np.max(self.plot_bounds, axis=0)

        #"meta-obstacle", "avoid","goal_lms","pattern","landmark_offset","search", "seed", and "pattern_offset"
        for obstacle in obstacles:  
            num_points = random.randint(2,20)  
            #print("initting {} obstacles of type: {}".format(num_points,obstacle))
            points = []
            while len(points) < num_points:
                # Generate random points within the bounding box
                random_points = np.random.rand(num_points, 2)
                random_points[:, 0] = random_points[:, 0] * (max_x - min_x) + min_x
                random_points[:, 1] = random_points[:, 1] * (max_y - min_y) + min_y
                
                # Check which points are inside the contour
                mask = contour.contains_points(random_points)
                points_inside = random_points[mask]
                
                # Add the valid points to the list
                points.extend(points_inside.tolist())
                
                # Limit the number of points to the desired number
                points = points[:num_points]

            self.obstacle_locations[obstacle] = points  

    def init_goals(self,goals): 
        contour = Path(self.plot_bounds)
        # Get the bounding box of the contour
        min_x, min_y = np.min(self.plot_bounds, axis=0)
        max_x, max_y = np.max(self.plot_bounds, axis=0)
    
        for target in goals: 
            num_points = random.randint(2,20)  
            #print("initting {} obstacles of type: {}".format(num_points,target))
            points = []
            while len(points) < num_points:
                # Generate random points within the bounding box
                random_points = np.random.rand(num_points, 2)
                random_points[:, 0] = random_points[:, 0] * (max_x - min_x) + min_x
                random_points[:, 1] = random_points[:, 1] * (max_y - min_y) + min_y
                
                # Check which points are inside the contour
                mask = contour.contains_points(random_points)
                points_inside = random_points[mask]
                
                # Add the valid points to the list
                points.extend(points_inside.tolist())
                
                # Limit the number of points to the desired number
                points = points[:num_points] 

            self.target_locations[target] = points  

    def init_constraints(self,constraints): 
        if "avoid" in constraints.keys(): 
            self.init_obstacles(constraints["avoid"])  
        
        if "goal_lms" in constraints.keys(): 
            self.init_goals(constraints["goal_lms"]) 

        if "search" in constraints.keys(): 
            self.init_goals(constraints["goal_lms"]) 

        self.simBot.gt_targets = self.target_locations
        self.simBot.gt_obstacles = self.obstacle_locations 

    def init_pose(self,max_distance=10): 
        path = Path(self.plot_bounds)
    
        # Get the bounding box of the contour
        min_x = min(self.plot_bounds[:,0]); max_x = max(self.plot_bounds[:,0]) 
        min_y = min(self.plot_bounds[:,1]); max_y = max(self.plot_bounds[:,1])   
        if np.random.rand() < 0.5: 
            print("initting robot inside plot bounds")
            #then its inside the plot bounds 
            point_inside = None
            while point_inside is None:
                # Generate a random point within the bounding box
                random_point = np.random.rand(1, 2)
                random_point[:, 0] = random_point[:, 0] * (max_x - min_x) + min_x
                random_point[:, 1] = random_point[:, 1] * (max_y - min_y) + min_y
                #print("random_point: ",random_point)
                if np.any(np.isnan(random_point)): 
                    raise OSError 
                # Check if the point is inside the contour
                if path.contains_point(random_point[0]):
                    point_inside = random_point[0] 
                    self.robot_traj[0,:2] = point_inside  
                    #pick random initial heading 
                    self.robot_traj[0,2] = random.uniform(0, 2*np.pi) 
                '''
                else:
                    print("path.constains_point(random_point[0]): ",path.contains_point(random_point[0]))
                '''
        else:
            print("initting robot outside plot bounds")
            #the its outside the plot bounds 
            point_outside = None
            while point_outside is None:
                # Generate a random point within an extended bounding box
                random_point = np.random.rand(1, 2)
                random_point[:, 0] = random_point[:, 0] * ((max_x + max_distance) - (min_x - max_distance)) + (min_x - max_distance)
                random_point[:, 1] = random_point[:, 1] * ((max_y + max_distance) - (min_y - max_distance)) + (min_y - max_distance) 
                #print("random_point: ",random_point) 
                if np.any(np.isnan(random_point)): 
                    raise OSError 
                # Check if the point is outside the contour
                if not path.contains_point(random_point[0]):
                    # Check if the point is within the max_distance from the contour
                    distances = np.linalg.norm(self.plot_bounds - random_point, axis=1) 
                    if np.min(distances) <= max_distance:
                        point_outside = random_point[0]
                        self.robot_traj[0, :2] = point_outside
                        self.robot_traj[0, 2] = random.uniform(0, 2 * np.pi)
                '''
                else:
                    print("path.contains_point(random_point[0]): ",path.contains_point(random_point[0]))
                '''

    def parse_next_step(self,policy):
        if not os.path.exists("parsed_steps"):
            os.mkdir("parsed_steps") 

        next_step = self.current_step + 1 
        step = get_step_from_policy(policy,next_step) 
         
        print("this is the next step: ",step) 
        print() 

        with open("prompts/step_parser_codeGen.txt","r") as f: 
            prompt = f.read() 
        
        enhanced_prompt = prompt.replace("*INSERT_QUERY*",self.prompt_path)
        enhanced_prompt = enhanced_prompt.replace("*INSERT_STEPS*",policy)
        enhanced_prompt = enhanced_prompt.replace("*INSERT_STEP*",step)

        code = generate_with_openai(enhanced_prompt) 
        clean_code = remove_chatGPT_commentary(code) 

        print("executing ... ") 

        local_scope = {}
        try: 
            exec(clean_code, {"self": self}, local_scope)
            with open("parsed_steps/step"+str(self.current_step)+".txt","w") as f:
                f.write(clean_code)
        except:
            if clean_code.find("import") == -1: 
                print("we were missing an import ... lets try again!") 

                with open("debug_clean_code.txt","w") as f:
                    f.write(clean_code) 

                clean_code = fix_imports(clean_code)

                with open("debug_clean_code-postFix.txt","w") as f:
                    f.write(clean_code) 

                try: 
                    exec(clean_code, {"self": self}, local_scope) 
                    with open("parsed_steps/step"+str(self.current_step)+".txt","w") as f:
                        f.write(clean_code)
                except:
                    print("ERROR: Could not execute this code ....")
                    print()
                    print(clean_code)
                    print()
                    input("Edit the code in corrected_code.py and press enter to continue ...") 
                    with open("corrected_code.py") as file: 
                        exec(file.read())  
                    shutil.copy("corrected_code.py","parsed_steps/step"+str(self.current_step)+".txt")
            else: 
                print("ERROR: Could not execute this code ....")
                print()
                print(clean_code)
                print()
                input("Edit the code in corrected_code.py and press enter to continue ...") 
                with open("corrected_code.py") as file: 
                    exec(file.read())  
                shutil.copy("corrected_code.py","parsed_steps/step"+str(self.current_step)+".txt")

        print()
        print("Done executing that step!") 

    def rehearse(self,policy_constraints): 
        #load in the polciy 
        print("reading the policy...")
        with open(self.final_policy_path,"r") as f:
            policy = f.read()

        self.policy_constraints = policy_constraints

        #for epochs in epoch
        #self.done = False 

        #instantiate obstacles and targets 
        print("initting constraints...")
        self.init_constraints(policy_constraints)

        self.init_pose() 
        self.simBot.traj_cache[0,:] = self.robot_traj[0,:] 

        self.current_step = 0 

        tstep = 0 
        self.plot_frame(tstep) 

        next_step = self.current_step + 1 
        current_step = get_step_from_policy(policy,next_step) 

        while current_step is not None: 
            print("entered the while loop") 
            self.parse_next_step(policy)   
            last_tstep = len(self.simBot.traj_cache) 
            print("last_tstep: ",last_tstep) 
            for t in range(tstep,last_tstep + 1):
                self.plot_frame(t) 
            tstep = last_tstep 
            print("Moving on to next step!")
            self.current_step += 1  
            current_step = get_step_from_policy(policy,self.current_step) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore Test")

    # Add an argument for test_query
    parser.add_argument(
        "--prompt_path",
        type=str,
        help="Path to desired prompt"
    )

    parser.add_argument(
        "--final_policy_path",
        type=str,
        help="Path to desired prompt"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration file",
        default="configs/example_config.toml",
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
    
    #THIS STUFF IS JUST FOR RUNNING REHEARSE POLICIES ON ITS OWN FOR DEBUGGING PURPOSES 

    #Plant in 1mx1m grid in the bounded area. Avoid driving over wherever you’ve planted or any conmods. 
    policy_constraints = {}
    policy_constraints["meta-obstacle"] = ["planted areas","conmods"] 
    policy_constraints["avoid"] = ["conmods"] 
    policy_constraints["seed"] = True  
    policy_constraints["pattern"] = "grid"     
    policy_constraints["pattern_offset"] = 1 

    #THIS STUFF IS JUST FOR RUNNING REHEARSE POLICIES ON ITS OWN FOR DEBUGGING PURPOSES 

    pR.rehearse(policy_constraints) #TO DO: FOR THE REAL DEAL ILL SAVE THE CONSTRAINTS SOMEWHERE AND LOAD THEM IN SO THIS FUNC WILL HAVE 0 ARGS 