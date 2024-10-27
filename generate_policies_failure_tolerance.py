from LLMGuidedSeeding_pkg.utils.llm_utils import generate_with_openai 
from LLMGuidedSeeding_pkg.utils.gen_utils import dictify 

import argparse
import numpy as np 
import os 
import toml 
import shutil 

from UI import ConversationalInterface
from LLMGuidedSeeding_pkg.utils.failure_tolerance_utils import get_unexecuted_steps, modify_constraints_for_failure, create_enhanced_prompt
class PolicyGeneratorFailure: 
    def __init__(
        self,
        prompt_path="example.txt",
        config_path="example.toml",
        logging_directory="logs",
        plot_bounds_path="random_path.csv",
        seed_gun_failure=True,  # New parameter
        object_label="cement",        # New parameter
        current_step= 5  # New parameter
    ):
        self.logging_directory = logging_directory 
        self.query = self.read(prompt_path)
        self.settings = toml.load(config_path)  
        self.plot_bounds = np.genfromtxt(plot_bounds_path)
        self.current_policy = self.load_policy()
        self.validPolicy = False 
        self.feedback= None 
        self.init_waypoint = None #this is a waypoint to get the robot within bounds if it's initially out of bounds 
        self.base_url = os.path.dirname(os.path.abspath(__file__))
        self.learned_objects = []
        #self.ConversationalInterface = ConversationalInterface()
        
        self.seed_gun_failure = seed_gun_failure  # Store the failure state
        self.object_label = object_label            # Store the object label
        self.current_step = current_step            # Store the current step


    def load_policy(self):
        """Load the current policy from the policy.txt file."""
        try:
            with open("policy.txt", "r") as f:
                content = f.read().strip()  # Read and strip whitespace
                return content if content else None  # Return None if the file is empty
        except FileNotFoundError:
            print("policy.txt not found. Initializing current_policy as None.")
            return None
        
    def read(self,file): 
        with open(file,"r") as f:
            return f.read() 

    def ask_object_clarification(self,object_name):
        print(f'I dont know what a {object_name} is. Can you write a description in OOD_obj_description.txt and press Enter to continue?')
        input("Press Enter to continue") 
        with open("OOD_obj_description.txt","r") as f:
            obj_description = f.read()
        filename = object_name.replace(" ", "_") + ".txt" 
        if not os.path.exists("./custom_obj_descriptions"):
            os.mkdir("./custom_obj_descriptions")
        shutil.copyfile("OOD_obj_description.txt",os.path.join("custom_obj_descriptions",filename)) 

    def ask_policy_verification(self,policy):
        print("What do you think of this policy? " + "\n" + policy) 
        input("Enter any feedback into feedback.txt (or leave it empty if you have no feedback) and press Enter to continue ...")        
        with open("feedback.txt","r") as f:
            self.feedback = f.read()
        if len(self.feedback) == 0:
            return True 
        else:
            return False 
        
    def verify_policy(self,policy): 
        #ask the user if they approve of this policy 
        #print("policy verification result:",self.conversational_interface.ask_policy_verification(policy))
        #print("policy: ",policy) 
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
            prompt_lms = {} 
            for k in constraints.keys(): 
                #print("constraints[k]: ",constraints[k]) 
                #print(type(constraints[k])) 
                if isinstance(constraints[k],str): 
                    if "and" in constraints[k]:
                        #print("And is in the constraints!") 
                        split_constraints = constraints[k].split("and") 
                        constraints_list = [split_constraints[0],split_constraints[1]] 
    
                        query = "Should '" +  split_constraints[0] + "' and '" + split_constraints[1] + """' be considered as separate objects? Or are these just adjectives to describe an object?
                        Return yes is they should be considered separately.""" 
                        response,_ = generate_with_openai(query) 
                        if "yes" in response.lower():
                            constraints[k] = constraints_list 

                    if "," in constraints[k]: 
                        split_constraint = constraints[k].split(",") 
                        #print("split_constraint: ",split_constraint)
                        constraints[k] = split_constraint 

                if isinstance(constraints[k],list):  
                    #print("this is  a list!")
                    if isinstance(constraints[k][0],str): 
                        if "," in constraints[k][0]:
                            split_constraint = constraints[k][0].split(",") 
                            #print("split_constraint: ",split_constraint)
                            constraints[k] = split_constraint 

            print("constraints: ",constraints) 

            for lm in prompt_lms:
                #is the lm in or out of distribution 
                query = "Would a " + lm + " be in-distribution for object detectors like yolo world? Yes or no."  
                response,_ = generate_with_openai(query) 
                if not "yes" in response.lower() and lm not in self.learned_objects:
                    print("I dont know what {} is. Ill have to ask.".format(lm))
                    #self.ask_object_clarification(lm) 
                    self.conversational_interface.ask_object_clarification(lm) 
                    #because the interface doesnt exist yet im just going to write the object descriptors and save them in txt files 
                    self.learned_objects.append(lm)  

            with open("prompts/get_policy_steps.txt","r") as f:
                prompt = f.read() 

            enhanced_prompt = prompt.replace("*INSERT_QUERY*",self.query)
            enhanced_prompt = enhanced_prompt.replace("*INSERT_CONSTRAINTS*",str(constraints))
            print("using enhanced prompt to generate a policy") 
            self.current_policy,_ = generate_with_openai(enhanced_prompt) 
            #self.verify_policy(self.current_policy)
        
        else:
            print("generating new policy because the seed gun failed.")
            if self.seed_gun_failure:
                # Handling the scenario where the seed gun has failed
                print(f"Handling failure with label: {self.object_label}")

                # Retrieve the original steps from the current policy
                original_steps = [step for step in self.current_policy.splitlines() if step.strip()]

                # Retrieve the unexecuted steps based on current_step
                unexecuted_steps = get_unexecuted_steps(self.current_step, original_steps)
                
                # Modify the constraints based on the failure situation
                constraints = modify_constraints_for_failure(constraints, self.object_label)
                print(constraints)
                # Generate a new policy based on the unexecuted steps and updated constraints
                print("Generating new policy based on unexecuted steps and modified constraints...")
                enhanced_prompt = create_enhanced_prompt(self.query, constraints, unexecuted_steps, original_steps)
                print(enhanced_prompt)
                self.current_policy, _ = generate_with_openai(enhanced_prompt)
                print(self.current_policy)
            
        '''
        else: 
            with open("prompts/modify_policy.txt","r") as f: 
                prompt = f.read() 

            print("constraints: ",constraints) 

            enhanced_prompt = prompt.replace("*INSERT_PROMPT*",self.query)
            enhanced_prompt = prompt.replace("*INSERT_CONSTRAINT_DICTIONARY*",str(constraints) ) 
            enhanced_prompt = prompt.replace("*INSERT_POLICY*",self.current_policy)
            enhanced_prompt = prompt.replace("*INSERT_FEEDBACK*",self.feedback)
            print("modifying policy...")
            modified_policy,_ = generate_with_openai(enhanced_prompt)
            print("modified_policy: ",modified_policy)
            self.current_policy = modified_policy 
            self.verify_policy(self.current_policy) 
        '''

    def parse_prompt(self): 
        print("parsing prompt to get constraints ...")
        with open("prompts/get_prompt_constraints.txt","r") as f:
            constraints_prompt = f.read()
        enhanced_query = constraints_prompt.replace("*INSERT_QUERY*",self.query)
        llm_result,_ = generate_with_openai(enhanced_query) 
        constraints = {} 
        i0 = llm_result.index("{"); i1 = llm_result.index("}")
        parsed_results = llm_result[i0:i1+1]
        if "}" not in parsed_results:
            parsed_results = parsed_results + "}"
        constraints = dictify(parsed_results) 
        print("constraints: ",constraints)
        return constraints
        
    def gen_policy(self): 
        print("identifying constraints ...") 
        #1. Identify constraints and goal landmarks from the prompt 
        constraints = self.parse_prompt()
        self.build_policy(constraints) 
        with open("policy.txt","w") as f:
            f.write(self.current_policy)

        print("Edit policy.txt until youre happy with it")
        input("Press Enter to Continue")         

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
    
    print("initting the Policy Generator with these arguments: ",args)

    pg = PolicyGeneratorFailure(
        prompt_path=args.prompt_path,
        config_path=args.config_path, 
        logging_directory=args.logging_dir,
        plot_bounds_path=args.plot_bounds_path
    )

    pg.gen_policy() 