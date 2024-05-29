from LLMGuidedSeeding_pkg import *

import argparse
import numpy as np 
import os 
import toml 
from UI_pkg import ConversationalInterface

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
        self.settings = toml.load(config_path)
        self.plot_bounds = np.genfromtxt(plot_bounds_path)
        self.current_policy = None 
        self.validPolicy = False 
        self.feedback= None 
        self.init_waypoint = None #this is a waypoint to get the robot within bounds if it's initially out of bounds 
        self.conversational_interface = ConversationalInterface()
        #print("self.settings: ",self.settings)
        with open(self.settings["commonObj_path"],"r") as f: 
            self.common_objects = [line.strip() for line in f]
        self.learned_objects = []
        self.policy_iters = 0 
        self.max_policy_iters = 5
         
    def read(self,file): 
        with open(file,"r") as f:
            return f.read() 

    def verify_policy(self,policy): 
        #ask the user if they approve of this policy 
        #print("policy verification result:",self.conversational_interface.ask_policy_verification(policy))
        print("Policy: ", policy)
        if self.conversational_interface.ask_policy_verification(policy):
            self.validPolicy = True 
            print("Found a valid policy approved by the human!")
            with open(os.path.join(self.logging_directory,"finalPolicy.txt"),"w") as f:
                f.write(policy)
        else: 
            print("Updating feedback!")
            self.feedback = self.conversational_interface.feedback

    def build_policy(self,constraints): 
        print("building policy...")
        if self.feedback is None and self.current_policy is None: 
            print("feedback is none!")
            #1. Navigate to goal points, respecting the constraints 
            # "avoid","goal_lms","pattern","landmark_offset","search", and "pattern_offset","seed" = True/False.
            # check all of these landmarks 
            prompt_lms = []
            if "goal_lms" in constraints.keys():
                if constraints["goal_lms"] is list:
                    prompt_lms.extend(constraints["goal_lms"])
                elif constraints["goal_lms"] is str: 
                    prompt_lms.append(constraints["goal_lms"])
            if "avoid" in constraints.keys(): 
                if constraints["avoid"] is list: 
                    prompt_lms.extend(constraints["avoid"])
                elif constraints["avoid"] is str:
                    prompt_lms.append(constraints["avoid"])

            for lm in prompt_lms:
                if not lm.lower() in self.common_objects and lm.lower() not in [x.name for x in self.learned_objects]:
                    print("I dont know what {} is. Ill have to ask.".format(lm))
                    self.conversational_interface.ask_object_clarification(lm) 
                    #because the interface doesnt exist yet im just going to write the object descriptors and save them in txt files 
                    self.learned_objects.append(lm)  

            #TO DO: ask object clarification 
            with open("prompts/get_policy_steps.txt","r") as f:
                prompt = f.read()
            enhanced_prompt = prompt.replace("*INSERT_QUERY*",self.query)
            enhanced_prompt = enhanced_prompt.replace("*INSERT_CONSTRAINTS*",str(constraints))
            self.current_policy = generate_with_openai(enhanced_prompt)
        else: 
            with open("prompts/modify_policy.txt","r") as f: 
                prompt = f.read() 
            enhanced_prompt = prompt.replace("*INSERT_PROMPT*",self.query)
            enhanced_prompt = prompt.replace("*INSERT_CONSTRAINT_DICTIONARY*",constraints)
            enhanced_prompt = prompt.replace("*INSERT_POLICY*",self.current_policy)
            enhanced_prompt = prompt.replace("*INSERT_FEEDBACK*",self.feedback)
            #print("this is the new prompt: ",enhanced_prompt)
            print("modifying policy...")
            modified_policy = generate_with_openai(enhanced_prompt)
            print("modified_policy: ",modified_policy)
            self.current_policy = modified_policy 

    def parse_prompt(self): 
        print("parsing prompt to get constraints ...")
        with open("prompts/get_prompt_constraints.txt","r") as f:
            constraints_prompt = f.read()
        enhanced_query = constraints_prompt.replace("*INSERT_QUERY*",self.query)
        llm_result = generate_with_openai(enhanced_query) 
        print("llm_result:",llm_result)
        '''
        #Debug
        with open("tmp.txt","w") as f:
            f.write(llm_result)
            f.close()
        '''       
        constraints = {} 
        #if "?" not in llm_result:
        i0 = llm_result.index("{"); i1 = llm_result.index("}")
        parsed_results = llm_result[i0:i1+1]
        constraints = dictify(parsed_results)
        
        return constraints
        
    def gen_policy(self): 
        #1. Identify constraints and goal landmarks from the prompt 
        constraints = self.parse_prompt()
        print("constraints: ",constraints)
        while not self.validPolicy:
            if self.policy_iters < self.max_policy_iters:
                #2. Come up with policy
                if self.policy_iters == 0:
                    policy = self.build_policy(constraints)
                #3. Verfiy with user 
                self.verify_policy(policy)
                #4. Integrate user feedback 
                if not self.validPolicy:
                    policy = self.build_policy(constraints) 
            else:
                raise Exception("Cannot come up with an acceptable policy :(")
            self.policy_iters += 1 

        code =  code_gen(self.policy)

        
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
    
    print("initting the Policy Generator with these arguments: ",args)

    pg = PolicyGenerator(
        prompt_path=args.prompt_path,
        config_path=args.config_path, 
        logging_directory=args.logging_dir,
        plot_bounds_path=args.plot_bounds_path
    )

    pg.gen_policy() 