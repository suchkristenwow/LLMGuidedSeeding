from LLMGuidedSeeding_pkg import *
import argparse
import numpy as np 
import os 

class PolicyGenerator: 
    def __init__(
        self,
        prompt_path="example.txt",
        config_path="example.toml",
        logging_directory="logs",
        plot_bounds_path="random_path.csv",
    ):
        self.logging_directory = logging_directory 
        self.query = self.read(prompt_path)
        self.settings = self.read(config_path)     
        self.plot_bounds = np.genfromtxt(plot_bounds_path)
        self.validPolicy = False 
        self.feedback_constraints = None 
        self.init_waypoint = None #this is a waypoint to get the robot within bounds if it's initially out of bounds 
        with open(self.settings["commonObj_path"],"r") as f: 
            self.common_objects = [line.strip() for line in f]
        self.learned_objects = []
        # ROBOT STUFF  
        odom_listener = OdometryListener(self.settings["odometry_topic"])

    def read(self,file): 
        with open(file,"r") as f:
            return f.read() 

    def verify_policy(self,policy): 
        #ask the user if they approve of this policy 
        if self.conversational_interface.ask_policy_verification(policy):
            self.validPolicy = True 
            print("Found a valid policy approved by the human!")
            with open(os.path.join(self.logging_directory,"finalPolicy.txt"),"w") as f:
                f.write(policy)

    def build_policy(self,constraints): 
        #1. query the current pose, if we're not in the bounds, plan to the nearest point in the bounds, avoiding
        #anything in the given constraints 
        policy_description = {}
        if self.feedback_constraints is None: 
            current_pose = self.odom_listener.get_pose()
            if not check_plot_bounds(current_pose,self.plot_bounds): 
                self.init_waypoint = get_closest_waypoint(current_pose,self.plot_bounds)
                policy_description[1] = "Navigate to the Desired Operation Area"
            #2. navigate to goal points, respecting the constraints 
            # "avoid","goal_lms","pattern","landmark_offset","search", and "pattern_offset","seed" = True/False.
            # check all of these landmarks 
            for lm in constraints["goal_lms"] + constraints["avoid"]:
                if not lm.lower() in self.common_objects and lm.lower() not in [x.name for x in self.learned_objects]:
                    self.learned_objects.append(self.conversational_interface.ask_object_clarification(lm))  
            with open("prompts/get_policy_steps.txt","r") as f:
                prompt = f.read()
            enhanced_prompt = prompt.replace("*INSERT_QUERY*",self.query)
            enhanced_prompt = enhanced_prompt.replace("*INSERT_CONSTRAINTS*",constraints)
            llm_result = generate_with_openai(enhanced_prompt)
            policy_description = parse_steps(llm_result,policy_description)
            return policy_description 
        else:
            modify_policy()
            return new_policy_description 

    def parse_prompt(self): 
        with open("prompts/get_prompt_constraints.txt","r") as f:
            constraints_prompt = f.read()
        enhanced_query = constraints_prompt.replace("*INSERT_QUERY*",self.query)
        llm_result = generate_with_openai(enhanced_query) 
        '''
        #Debug
        with open("tmp.txt","w") as f:
            f.write(llm_result)
            f.close()
        ''' 
        constraints = {} 
        if "?" not in llm_result:
            i0 = llm_result.index("{"); i1 = llm_result.index("}")
            parsed_results = llm_result[i0:i1+1]
            constraints = dictify(parsed_results)
        else:
            #TO DO 
            raise OSError 
        return constraints
        
    def gen_policy(self): 
        #1. Identify constraints and goal landmarks from the prompt 
        constraints = self.parse_prompt()
        while not self.validPolicy:
            if self.policy_iters < self.max_policy_iters:
                #2. Come up with policy
                policy = self.build_policy(constraints)
                #3. Verfiy with user 
                self.verify_policy(policy)
                #4. Integrate user feedback 
                if not self.validPolicy:
                    self.feedback_constraints = self.parse_feedback 
            else:
                raise Exception("Cannot come up with an acceptable policy :(")
        
if __name__ == "__main__":
    # Create an argument parser
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

    pg = PolicyGenerator(
        prompt_path=args.prompt_path,
        config_path=args.config_path, 
        logging_directory=args.logging_dir,
        plot_bounds_path=args.plot_bounds_path
    )

    while not pg.validPolicy :
        pg.gen_policy() 
    
    with open("policy.txt","w") as f:
        f.write(pg.policy)