from LLMGuidedSeeding_pkg.utils.llm_utils import generate_with_openai 
from LLMGuidedSeeding_pkg.utils.gen_utils import dictify 

import argparse
import numpy as np 
import os 
import toml 
import shutil 

from UI import ConversationalInterface

def clean_json_string(json_string):
    """
    Cleans a JSON-like string by removing comments and ensuring proper formatting.
    Ensures commas are correctly placed between key-value pairs without adding trailing commas.
    """
    lines = json_string.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.split("#")[0].split("//")[0].strip()  # Remove comments
        if line:
            cleaned_lines.append(line)

    # Join lines into a single string
    cleaned_json = "\n".join(cleaned_lines)

    # Replace Python-style True/False/None with JSON-compliant true/false/null
    cleaned_json = cleaned_json.replace("True", "true").replace("False", "false").replace("None", "null")

    # Remove any trailing commas inside JSON objects or arrays
    cleaned_json = cleaned_json.replace(",\n}", "\n}")
    cleaned_json = cleaned_json.replace(",\n]", "\n]")

    return cleaned_json


class PolicyGenerator: 
    def __init__(
        self,
        prompt_path="example.txt",
        config_path="example.toml",
        logging_directory="logs"
    ):
        self.prompt_name = os.path.basename(prompt_path)[:-4]
        print("prompt_name: ",self.prompt_name)
        self.logging_directory = logging_directory 
        self.query = self.read(prompt_path)
        self.settings = toml.load(config_path)  
        #self.plot_bounds = np.genfromtxt(plot_bounds_path)
        self.current_policy = None 
        self.validPolicy = False 
        self.feedback= None 
        self.init_waypoint = None #this is a waypoint to get the robot within bounds if it's initially out of bounds 
        self.base_url = os.path.dirname(os.path.abspath(__file__))
        self.learned_objects = []
        #self.ConversationalInterface = ConversationalInterface()

    def read(self,file): 
        with open(file,"r") as f:
            return f.read() 

    def ask_object_clarification(self,object_name):
        if not os.path.exists("./custom_obj_descriptions"):
            os.mkdir("./custom_obj_descriptions")

        object_name = object_name.replace(" ","_") 
        filename = object_name + ".txt"

        if not os.path.exists(os.path.join("custom_obj_descriptions",filename)): 
            print(f'I dont know what a {object_name} is. Can you write a description in OOD_obj_description.txt and press Enter to continue?')
            input("Press Enter to continue") 
            '''
            with open("OOD_obj_description.txt","r") as f:
                obj_description = f.read()
            '''
        
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

    def build_policy(self,constraints,dir_,iter): 
        print("building policy...")
        if self.feedback is None and self.current_policy is None: 
            prompt_lms = []
            for k in constraints.keys(): 
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
                    if isinstance(constraints[k][0],str): 
                        if "," in constraints[k][0]:
                            split_constraint = constraints[k][0].split(",") 
                            #print("split_constraint: ",split_constraint)
                            constraints[k] = split_constraint 

            #"meta-obstacle", "avoid","goal_lms","pattern","landmark_offset","search", "seed", and "pattern_offset". 
            if "avoid" in constraints.keys():
                if isinstance(constraints["avoid"],list): 
                    prompt_lms.extend(constraints["avoid"]) 
                elif isinstance(constraints["avoid"],str): 
                    prompt_lms.append(constraints["avoid"])
            if "goal_lms" in constraints.keys():
                if isinstance(constraints["goal_lms"],list): 
                    prompt_lms.extend(constraints["goal_lms"]) 
                elif isinstance(constraints["goal_lms"],str): 
                    prompt_lms.append(constraints["goal_lms"]) 
            if "search" in constraints.keys(): 
                if isinstance(constraints["search"],list): 
                    prompt_lms.extend(constraints["search"]) 
                elif isinstance(constraints["search"],str): 
                    prompt_lms.append(constraints["search"])  

            '''
            for lm in prompt_lms:   
                #is the lm in or out of distribution 
                query = "Would a " + lm + " be in-distribution for object detectors like yolo world? Specifically in the context of this prompt: " + self.query + "\n" + \
                    "For ease of parsing, please answer with a Yes or No."
                
                print("checking if {} is out of distribution".format(lm)) 
                response,_ = generate_with_openai(query) 
                print("response: ",response) 

                if not "yes" in response.lower() and lm not in self.learned_objects:
                    print("I dont know what {} is. Ill have to ask.".format(lm))
                    #self.ask_object_clarification(lm) 
                    #self.conversational_interface.ask_object_clarification(lm) 
                    self.ask_object_clarification(lm)
                    #because the interface doesnt exist yet im just going to write the object descriptors and save them in txt files 
                    self.learned_objects.append(lm)   
                    if 'flag' in lm:
                        with open(os.path.join(dir_,"ood_detection"+str(iter)+".txt"),"w") as f: 
                            f.write("True")
                else:
                    print(f"I think that {lm} is in-distribution") 
                    if 'flag' in lm: 
                        with open(os.path.join(dir_,"ood_detection"+str(iter)+".txt"),"w") as f:
                            f.write("False")
                
            '''
            
            with open("prompts/get_policy_steps.txt","r") as f:
                prompt = f.read() 

            enhanced_prompt = prompt.replace("*INSERT_QUERY*",self.query)
            enhanced_prompt = enhanced_prompt.replace("*INSERT_CONSTRAINTS*",str(constraints))
            print("using enhanced prompt to generate a policy") 
            self.current_policy,_ = generate_with_openai(enhanced_prompt) 
            #self.verify_policy(self.current_policy)
            
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

        llm_result,history = generate_with_openai(enhanced_query) 
        print("response: ",llm_result) 
        print() 

        constraints = {} 
        i0 = llm_result.index("{"); i1 = llm_result.index("}")
        parsed_results = llm_result[i0:i1+1]
        if "}" not in parsed_results:
            parsed_results = parsed_results + "}"
        constraints = dictify(parsed_results) 
        if "seed" in constraints.keys(): 
            if constraints["seed"]: 
                prompt = "Are you sure the user specified that they wanted you to plant seeds in the original prompt?" 
                print("double checking seeding requirement ...")
                response,_ = generate_with_openai(prompt,conversation_history=history) 
                if not "yes" in response.lower():
                    constraints["seed"] = False 
                
        print("constraints: ",constraints)

        return constraints
    
    def gen_policy(self, constraints = None, path = None, dir_ = None, iter_ = None):
        if constraints is None:  
            print("identifying constraints ...") 
            #1. Identify constraints and goal landmarks from the prompt 
            constraints = self.parse_prompt()
        
        print("building policy ...")
        self.build_policy(constraints,dir_,iter_) 

        if path is None: 
            with open("policy.txt","w") as f:
                f.write(self.current_policy)   
                f.close() 
        else: 
            if not os.path.exists(path): 
                with open(path,"w") as f: 
                    print("self.current_policy: ",self.current_policy)
                    print("writing policy to path: {}".format(path))
                    f.write(self.current_policy) 
                    f.close() 

    def code_gen(self,iter=None): 
        if not os.path.exists(os.path.join("thesis_experiments",self.prompt_name)):
            os.mkdir(os.path.join("thesis_experiments",self.prompt_name))

        prompt_dir = os.path.join("thesis_experiments",self.prompt_name) 

        constraint_dict = self.parse_prompt()
        with open(os.path.join(prompt_dir,"constraints_" + str(iter) + ".txt"),"w") as f: 
            f.write(str(constraint_dict))  
            f.close()  

        #regenerate policy 
        self.current_policy = None 

        if iter is not None:             
            policy_path = os.path.join(prompt_dir,"policy"+str(iter)+".txt")
            print("policy_path: ",policy_path) 
            self.gen_policy(constraints=constraint_dict,path=policy_path,dir_=prompt_dir,iter_=iter)
            final_policy = self.read(policy_path) 
        else: 
            self.gen_policy(constraints=constraint_dict)
            final_policy = self.read("policy.txt")

        prompt = self.read("prompts/step_parser_codeGen.txt")
        
        code_gen_prompt = prompt.replace("*INSERT_QUERY", self.query) 
        code_gen_prompt = code_gen_prompt.replace("*INSERT_CONSTRAINT_DICT*",str(constraint_dict))
        code_gen_prompt = code_gen_prompt.replace("*INSERT_POLICY*", final_policy)

        if len(self.learned_objects) > 0:
            for custom_obj in self.learned_objects: 
                filename  = custom_obj.replace(" ","_") + ".txt"
                with open("./custom_obj_descriptions/"+ filename,"r") as f:
                    obj_description = f.read() 
                code_gen_prompt += "\n" + f"The user has defined {custom_obj} like this: " + "\n" + obj_description 

        llm_result, _ = generate_with_openai(code_gen_prompt)

        if not iter is None: 
            with open(os.path.join(prompt_dir,"result" + str(iter) + ".txt"),"w") as f: 
                f.write(llm_result)
            f.close() 

    def seeding_failure_reGen(self,iter=None,image_path=None): 
        if not os.path.exists(os.path.join("thesis_experiments",self.prompt_name)):
            os.mkdir(os.path.join("thesis_experiments",self.prompt_name))

        prompt_dir = os.path.join("thesis_experiments",self.prompt_name) 
        print("prompt dir: ",prompt_dir) 

        constraint_dict = self.parse_prompt()
        with open(os.path.join(prompt_dir,"constraints_" + str(iter) + ".txt"),"w") as f: 
            f.write(str(constraint_dict))  
            f.close()  

        #regenerate policy 
        self.current_policy = None 

        if iter is not None:             
            policy_path = os.path.join(prompt_dir,"policy"+str(iter)+".txt")
            print("policy_path: ",policy_path) 
            self.gen_policy(constraints=constraint_dict,path=policy_path,dir_=prompt_dir,iter_=iter)
            final_policy = self.read(policy_path) 
        else: 
            raise OSError 

        prompt = self.read("prompts/fix_constraints.txt")
    
        old_code = self.read(os.path.join(prompt_dir,"result"+str(iter)+".txt"))

        code_gen_prompt = prompt.replace("*INSERT_QUERY", self.query) 
        code_gen_prompt = code_gen_prompt.replace("*INSERT_CONSTRAINT_DICT*",str(constraint_dict))
        code_gen_prompt = code_gen_prompt.replace("*INSERT_POLICY*", final_policy)

        if len(self.learned_objects) > 0:
            for custom_obj in self.learned_objects: 
                filename  = custom_obj.replace(" ","_") + ".txt"
                with open("./custom_obj_descriptions/"+ filename,"r") as f:
                    obj_description = f.read() 
                code_gen_prompt += "\n" + f"The user has defined {custom_obj} like this: " + "\n" + obj_description 

        print("prompt: ",code_gen_prompt) 

        llm_result, history = generate_with_openai(code_gen_prompt,image_path=image_path)

        print("llm_result: ",llm_result)

        print("constraint_dict: ",constraint_dict) 

        print() 

        new_constraints = {} 
        i0 = llm_result.index("{"); i1 = llm_result.index("}")
        parsed_results = llm_result[i0:i1+1]
        if "}" not in parsed_results:
            parsed_results = parsed_results + "}"

        new_constraints = clean_json_string(parsed_results)                 

        prompt = self.read("prompts/seeding_failure_prompt.txt") 

        code_gen_prompt = prompt.replace("*INSERT_CONSTRAINT_DICT*",str(new_constraints))
        code_gen_prompt = code_gen_prompt.replace("*INSERT_OLD_CODE*",old_code)     

        print("prompt: ",code_gen_prompt)    
        
        llm_result, history = generate_with_openai(code_gen_prompt,image_path=image_path,conversation_history=history)  

        print("llm_result: ",llm_result) 
        print() 

        if not ">>>" in  llm_result:
            prompt = "Can you re-write the code such that it fixes the error caused by the seeder hitting something hard?" 
            print("prompt: ",prompt) 
            llm_result, _ = generate_with_openai(prompt,image_path=image_path,conversation_history=history)  
            print("second response: ",llm_result) 

        if not os.path.exists(os.path.join(prompt_dir,"faultRecovery")):
            os.mkdir(os.path.join(prompt_dir,"faultRecovery"))

        if not iter is None: 
            with open(os.path.join(prompt_dir,"faultRecovery/result" + str(iter) + ".txt"),"w") as f: 
                print("writing {}".format(os.path.join(prompt_dir,"faultRecovery/result" + str(iter) + ".txt")))
                f.write(llm_result)
            f.close() 
            with open(os.path.join(prompt_dir,"faultRecovery/new_constraints" + str(iter) + ".txt"),"w") as f: 
                print("writing {}".format(os.path.join(prompt_dir,"faultRecovery/new_constraints" + str(iter) + ".txt")))
                f.write(str(new_constraints))
            f.close() 

                
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

    '''
    parser.add_argument(
        "--plot_bounds_path",
        type=str,
        help="Path of a csv file containing the perimeter of the desired area of operation",
        default="random_path.csv",
    )
    '''
    
    # Parse the command-line arguments
    args = parser.parse_args() 
    
    print("initting the Policy Generator with these arguments: ",args)

    pg = PolicyGenerator(
        prompt_path=args.prompt_path,
        config_path=args.config_path, 
        logging_directory=args.logging_dir
    )
    pg.gen_policy()
    #pg.seeding_failure_reGen(iter=0,image_path="/home/kristen/Downloads/sidewalk.jpg")