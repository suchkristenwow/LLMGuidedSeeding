from LLMGuidedSeeding_pkg.utils.rehearsal_utils import * 
from UI import ConversationalInterface 
import os 
import toml 
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import random 
from matplotlib.path import Path 
import pickle 
from shapely.geometry import Polygon, Point 
from LLMGuidedSeeding_pkg.robot_client.robot import Robot 
from LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms 
from LLMGuidedSeeding_pkg.utils.codeGen_utils import * 
from chatGPT_written_utils import *  
import math 


class policyRehearsal: 
    def __init__(self,args):
        self.final_policy_path = args.final_policy_path

        with open(args.prompt_path,'r') as f:
            self.query = f.read()

        self.logging_dir = args.logging_dir  


        with open(args.constraints_dict_path,"rb") as handle: 
            self.policy_constraints = pickle.load(handle) 

        '''
        self.plot_bounds = np.genfromtxt(args.plot_bounds_path,delimiter=",")  
        self.plot_bounds = self.plot_bounds[~np.isnan(self.plot_bounds).any(axis=1)]
        if not np.array_equal(self.plot_bounds[0], self.plot_bounds[-1]):
            self.plot_bounds = np.vstack([self.plot_bounds, self.plot_bounds[0]]) 
        '''
        self.config_path = args.config_path
        self.parameters = toml.load(args.config_path) 
        
        self.obstacle_objs = {} 
        self.target_objs = {}

        self.meta_constraints = {} 
        
        with open("./configs/std_imports.txt","r") as f:
            self.std_imports = f.read()
            
        self.current_step = 0 

        self.robot_transforms = robotTransforms(self.config_path)

        #TODO: UI ... idk how to integrate this, going to use command line for now 
        #self.conversational_interface = ConversationalInterface() 

        self.feedback = None 
        self.obstacle_descriptions = {} #keys are labels, subkeys are size and shape and description 
        self.target_descriptions = {} 

        self.existing_landmarks = [] 

        self.clip_model = CLIPModel() 

        self.max_exec_tries = 3

    def ask_human(self,question): 
        print("Question: ",question)
        print()
        input("Type your answer in feedback.txt and press Enter to Continue")
        with open("feedback.txt","r") as f:
            self.feedback = f.read()

    def init_obstacles(self,obstacles,lm_type,max_obstacles=20):  
        """
        n_obstacles = random.randint(2,max_obstacles)  
        for obstacle in obstacles: 
            self.obstacle_objs[obstacle] = [] 
            if self.clip_model.in_distribution(obstacle):
                print("obstacle: {} is in distribution".format(obstacle))
                with open("prompts/shape_prompt.txt","r") as f: 
                    prompt = f.read() 
                enhanced_prompt = prompt.replace("*INSERT_OBJECT*",obstacle)
                enhanced_prompt = enhanced_prompt.replace("*INSERT_DESCRIPTION*","") 
                enhanced_prompt = enhanced_prompt.replace('*INSERT_IMPORTS*',self.std_imports)
            else:
                #save this object for later 
                if not os.path.exists("prompts/custom_objects"):
                    os.mkdir("prompts/custom_objects") 

                with open("prompts/human_description_shape_prompt.txt",'r') as f:
                    prompt = f.read()

                if obstacle in [x[:-4] for x in os.listdir("prompts/custom_objects")]:
                    with open("prompts/custom_objects/"+obstacle+".txt","r") as f:
                        description = f.read()

                    enhanced_prompt = prompt.replace("*INSERT_OBJECT*",obstacle)
                    desc_insert = build_description_prompt(description,obstacle)
                    enhanced_prompt = enhanced_prompt.replace("*INSERT_DESCRIPTION*",desc_insert) 
                    enhanced_prompt = enhanced_prompt.replace('*INSERT_IMPORTS*',self.std_imports)
                else: 
                    print("these are the custom objects: ", [x[:-4] for x in os.listdir("prompts/custom_objects")])
                    print("obstacle: {} is out of distribution".format(obstacle)) 

                    with open("prompts/object_clarification_prompt.txt","r") as f:
                        question_prompt = f.read()
                    question = question_prompt.replace("*INSERT_OBJECT*",obstacle) 
                    self.ask_human(question) 

                    enhanced_prompt = prompt.replace("*INSERT_OBJECT*",obstacle)
                    desc_insert = build_description_prompt(self.feedback,obstacle)
                    enhanced_prompt = enhanced_prompt.replace("*INSERT_DESCRIPTION*",desc_insert) 
                    enhanced_prompt = enhanced_prompt.replace('*INSERT_IMPORTS*',self.std_imports)

                    print("writing {} ....".format("prompts/custom_objects/"+obstacle+".txt"))
                    with open("prompts/custom_objects/"+obstacle+".txt","w") as f:
                        f.write(self.feedback) 

            n_obstacles = random.randint(2,max_obstacles) 

            raw_code,_ = generate_with_openai(enhanced_prompt)

            code = remove_chatGPT_commentary(raw_code)

            code = ensure_imports(code,self.std_imports)
        """
        
        for obstacle in obstacles: 
            self.obstacle_objs[obstacle] = [] 
            n_obstacles = random.randint(2,max_obstacles)  
            local_scope = {}
            local_scope.update({
                'plot_bounds': self.plot_bounds,
                'n_obstacles': n_obstacles,
                'existing_landmarks': self.existing_landmarks,
            })

            global_scope = globals().copy()  # Start with a copy of the current global scope

            local_scope = {}
            local_scope.update({
                'plot_bounds': self.plot_bounds,
                'n_obstacles': n_obstacles,
                'existing_landmarks': self.existing_landmarks,
            }) 

            global_scope.update(local_scope)
            
            with open("self_critique_logs/inital_attempt_init_obstacles.py","r") as f:
                code = f.read()

            '''
            with open("self_critique_logs/inital_attempt_init_obstacles.py",'w') as f:
                print("writing {}".format("inital_attempt_init_obstacles.py")) 
                f.write(code)   
            '''

            exec(compile(code, 'Codex', 'exec'),global_scope)  

            """
            try: 
                print("first try!") 
                exec(compile(code, 'Codex', 'exec'),global_scope)  

            except Exception as e:
                print("initial attempt failed: ",str(e)) 
                inserts = {} 
                if obstacle in [x[:-4] for x in os.listdir("prompts/custom_objects")]:
                    with open("prompts/custom_objects/"+obstacle+".txt","r") as f:
                        obstacle_description = f.read()  
                        desc_insert = build_description_prompt(description,obstacle_description) 
                else:
                    desc_insert = ""

                inserts["*INSERT_DESCRIPTION*"] = desc_insert
                inserts["*INSERT_OBJECT*"] = obstacle 
                inserts["*INSERT_IMPORTS*"] = self.std_imports 
                inserts["*INSERT_CODE*"] = code 
                inserts["*INSERT_ERROR*"] = str(e) 

                with open("prompts/salvage_prompt.txt","r") as f: 
                    critique_prompt =f.read()

                print("entering self critique mode")
                #task_name,prompt,insert_dict,local_scope,global_scope,std_imports,max_attempts 
                local_scope = self_critique_code("init_obstacles",critique_prompt,inserts,local_scope,global_scope,self.std_imports,self.max_exec_tries) 
            """
            
            if 'obstacles' in global_scope.keys():
                if lm_type == "obstacle": 
                    self.obstacle_objs[obstacle].extend(global_scope['obstacles']) 
                elif lm_type == "target": 
                    self.target_objs[obstacle].extend(global_scope['obstacles'])  

                self.existing_landmarks.extend(global_scope['obstacles']) 
                
            else: 
                if lm_type == "obstacle": 
                    self.obstacle_objs[obstacle].extend(local_scope['obstacles']) 
                elif lm_type == "target": 
                    self.target_objs[obstacle].extend(local_scope['obstacles'])  
                else:
                    raise OSError 
            
                self.existing_landmarks.extend(local_scope['obstacles']) 
            
            print("Successfully initted {}".format(obstacle))  
            print()  

    def init_constraints(self,constraints): 
        print("these are the constraints: ",constraints)

        if "avoid" in constraints.keys(): 
            self.init_obstacles(constraints["avoid"],'obstacle')  
            print("Done initting obstacles!") 

        if "goal_lms" in constraints.keys(): 
            self.init_obstacles(constraints["avoid"],'target')  
            print("Done initting goal lms") 

        if "search" in constraints.keys(): 
            self.init_obstacles(constraints["avoid"],'target')  
            print("Done initting search things")

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

        """
        with open("prompts/step_parser_codeGen.txt","r") as f: 
            prompt = f.read() 

        enhanced_prompt = prompt.replace("*INSERT_QUERY*",self.query)
        enhanced_prompt = enhanced_prompt.replace("*INSERT_POLICY*",policy)

        raw_code,_ = generate_with_openai(enhanced_prompt)

        code = remove_chatGPT_commentary(raw_code)
        """
        
        global_scope = globals().copy()  # Start with a copy of the current global scope 

        local_scope = {}

        local_scope.update({
            'config_path': self.config_path, 
            'plot_bounds': self.plot_bounds, 
            'init_pose': initted_pose, 
            'targets': self.target_objs, 
            'obstacles': self.obstacle_objs
        })

        global_scope.update(local_scope)

        with open("self_critique_logs/initial_attempt_policy_execution.py","r") as f:
            code = f.read() 
            
        exec(compile(code, 'Codex', 'exec'),global_scope)  

        """
        with open("self_critique_logs/initial_attempt_policy_execution.py","w") as f:
            f.write(code) 

        print("trying to execute: ","initial_attempt_policy_execution.py")

        try: 
            exec(compile(code, 'Codex', 'exec'),global_scope)  
        except Exception as e: 
            inserts = {} 
            custom_objs = []
            for obstacle in self.obstacle_objs:
                if obstacle in [x[:-4] for x in os.listdir("prompts/custom_objects")]:
                    custom_objs.append(obstacle) 
            for obstacle in self.target_objs:
                if obstacle in [x[:-4] for x in os.listdir("prompts/custom_objects")]:
                    custom_objs.append(obstacle)  
            
            desc_insert = "The user has provided a description of the following objects which are important to completing this task: " + "\n" 
            for obj in custom_objs: 
                with open("prompts/custom_objects/"+obj+".txt","r") as f:
                    obj_description = f.read() 
                desc_insert += obj_description + "\n"

            inserts['*INSERT_QUERY*'] = self.query
            inserts["*INSERT_DESCRIPTION*"] = desc_insert
            inserts["*INSERT_POLICY*"] = policy 
            inserts["*INSERT_IMPORTS*"] = self.std_imports 
            inserts["*INSERT_CODE*"] = code 
            inserts["*INSERT_ERROR*"] = str(e) 

            with open("prompts/salvage_policy_execution.txt","r") as f: 
                critique_prompt =f.read()

            print("entering self critique mode") 
            #task_name,prompt,insert_dict,local_scope,global_scope,std_imports,max_attempts 
            local_scope = self_critique_code("policy_execution",critique_prompt,inserts,local_scope,global_scope,self.std_imports,self.max_exec_tries) 
        """
        
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

    # Parse the command-line arguments
    args = parser.parse_args()  

    pR = policyRehearsal(args)
    
    #pR.test_existing_code("attempted_policy.py")
    pR.attempt_policy_execution() 