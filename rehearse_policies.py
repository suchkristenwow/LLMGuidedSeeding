from LLMGuidedSeeding_pkg import * 
from LLMGuidedSeeding_pkg.utils.rehearsal_utils import * 
from UI import ConversationalInterface 
import os 
import toml 
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import random 
from matplotlib.path import Path 
import pandas as pd 
import shutil 
import pickle 

class policyRehearsal: 
    def __init__(self,args):
        self.final_policy_path = args.final_policy_path
        self.prompt_path = args.prompt_path  
        self.logging_dir = args.logging_dir  
        with open(args.constraints_dict_path,"rb") as handle: 
            self.policy_constraints = pickle.load(handle) 

        self.plot_bounds = np.genfromtxt(args.plot_bounds_path,delimiter=",")  
        self.plot_bounds = self.plot_bounds[~np.isnan(self.plot_bounds).any(axis=1)]
        if not np.array_equal(self.plot_bounds[0], self.plot_bounds[-1]):
            self.plot_bounds = np.vstack([self.plot_bounds, self.plot_bounds[0]]) 

        self.config_path = args.config_path
        self.parameters = toml.load(args.config_path) 
        
        self.obstacle_locations = {} 
        self.target_locations = {} 
        self.meta_constraints = {} 
        
        with open("./configs/std_imports.txt","r") as f:
            self.std_imports = f.read()

        self.current_step = 0 

        self.robot_transforms = robotTransforms(self.config_path)

        #TODO: UI ... idk how to integrate this, going to use command line for now 
        #self.conversational_interface = ConversationalInterface() 

    def init_obstacles(self,obstacles,max_obstacles=15):  
        contour = Path(self.plot_bounds)
        # Get the bounding box of the contour
        min_x, min_y = np.min(self.plot_bounds, axis=0)
        max_x, max_y = np.max(self.plot_bounds, axis=0)

        #"meta-obstacle", "avoid","goal_lms","pattern","landmark_offset","search", "seed", and "pattern_offset"
        for obstacle in obstacles:  
            num_points = random.randint(2,max_obstacles)  
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

    def init_goals(self,goals,max_goals=15): 
        print("initalizing targets!")
        contour = Path(self.plot_bounds)
        # Get the bounding box of the contour
        min_x, min_y = np.min(self.plot_bounds, axis=0)
        max_x, max_y = np.max(self.plot_bounds, axis=0)
        
        for target in goals: 
            num_points = random.randint(2,max_goals)  
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
        print("these are the constraints: ",constraints)

        if "avoid" in constraints.keys(): 
            self.init_obstacles(constraints["avoid"])  
        
        if "goal_lms" in constraints.keys(): 
            self.init_goals(constraints["goal_lms"]) 

        if "search" in constraints.keys(): 
            self.init_goals(constraints["goal_lms"]) 

    def init_pose(self,max_distance=10): 
        path = Path(self.plot_bounds)
        init_pose = np.zeros((6,))
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
                    init_pose[:2] = point_inside  
                    #pick random initial heading 
                    init_pose[-1] = random.uniform(0, 2*np.pi) 
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
                        init_pose[:2]= point_outside
                        init_pose[-1] = random.uniform(0, 2 * np.pi)
                '''
                else:
                    print("path.contains_point(random_point[0]): ",path.contains_point(random_point[0]))
                '''

        return init_pose 

    def parse_next_step(self,policy):
        if not os.path.exists("parsed_steps"):
            os.mkdir("parsed_steps") 

        next_step = self.current_step + 1 
        step = get_step_from_policy(policy,next_step) 
        
        return step 
    
    def attempt_policy_execution(self): 
        #load in the polciy 
        print("reading the policy...")
        with open(self.final_policy_path,"r") as f:
            policy = f.read()

        print("initting constraints...")
        self.init_constraints(self.policy_constraints)
    
        initted_pose = self.init_pose() 

        with open("prompts/step_parser_codeGen.txt","r") as f: 
            prompt = f.read() 

        enhanced_prompt = prompt.replace("*INSERT_QUERY*",self.prompt_path)
        enhanced_prompt = enhanced_prompt.replace("*INSERT_POLICY*",policy)

        raw_code = generate_with_openai(enhanced_prompt) 
        with open("rawCode.txt","w") as f:
            f.write(raw_code) 

        clean_code = remove_chatGPT_commentary(raw_code)
        with open("cleaned_code.txt","w") as f: 
            f.write(clean_code)

        code = ensure_imports(clean_code,self.std_imports)
        with open("post-ensure_imports.txt","w") as f: 
            f.write(code)
            
        local_scope = {} 

        try:
            with open("attempted_policy.txt",'w') as f:
                f.write(code)  
            exec(code, {'config_path': self.config_path, 'plot_bounds': self.plot_bounds, 'init_pose': initted_pose, 'target_locations': self.target_locations, 'obstacle_locations': self.obstacle_locations}, local_scope)
        
        except Exception as e: 
            print("An error occurred during the execution of the code :(") 
            print(e)
            raise OSError 

    def test_existing_code(self,code_path):
        '''
        This function is to test the non-code gen parts 
        '''
        #instantiate obstacles and targets 
        print("initting constraints...")
        self.init_constraints(self.policy_constraints)

        initted_pose = self.init_pose() 
        print("initted_pose:",initted_pose)  

        local_scope = {}

        with open(code_path) as file:
            code = file.read()

        code = ensure_imports(code,self.std_imports)

        # Execute the code directly
        exec(code, {'config_path': self.config_path, 'plot_bounds': self.plot_bounds, 'init_pose': initted_pose, 'target_locations': self.target_locations, 'obstacle_locations': self.obstacle_locations}, local_scope)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore Test")

    # Add an argument for test_query
    parser.add_argument(
        "--prompt_path",
        type=str,
        help="Path to desired prompt"
    )

    parser.add_argument(
        "--constraints_dict_path",
        type=str,
        help="Path to the constraint dict from the gen_policies",
        default="random_constraints.pickle" 
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
    
    pR.attempt_policy_execution() 