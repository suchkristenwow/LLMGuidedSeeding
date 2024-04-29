from  import

class PolicyGenerator: 
    def __init__(
        self,
        query,
        config_path="example.toml",
        logging_directory="logs",
    ):
        self.logging_directory = logging_directory 
        self.query = query 
        self.settings = self.read_config(config_path)     
        self.validPolicy = False 
        self.feedback_constraints = None 

    '''
    def parse_feedback(self): 

    def verify_policy(self): 
        #ask the user if they approve of this policy 
        
    def build_policy(self): 
        return policy 
    '''

    def parse_prompt(self): 
        with open("prompts/get_prompt_constraints.txt","r") as f:
            constraints_prompt = f.read()
        enhanced_query = f.replace("*INSERT_QUERY*",self.query)
        constraints_result = generate_with_openai(enhanced_query) 
        print("constraints_result: ",constraints_result)
        #return constraints, goal_lms 
        raise OSError 
    
    def gen_policy(self): 
        #1. Identify constraints and goal landmarks from the prompt 
        constraints,goal_lms = self.parse_prompt()
        '''
        while not self.validPolicy:
            if self.policy_iters < self.max_policy_iters:
                #2. Come up with policy
                policy = self.build_policy(constraints,goal_lms)
                #3. Verfiy with user 
                self.verify_policy(policy)
                #4. Integrate user feedback 
                if not self.validPolicy:
                    self.feedback_constraints = self.parse_feedback 
            else:
                raise Exception("Cannot come up with an acceptable policy :(")
        '''
        
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Explore Test")

    # Add an argument for test_query
    parser.add_argument(
        "--prompt",
        type=str,
        help="User prompt"
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

    # Parse the command-line arguments
    args = parser.parse_args() 

    pg = PolicyGenerator(
        prompt=args.prompt,
        config_path=args.config_path, 
        logging_directory=args.logging_dir
    )

    while not pg.validPolicy :
        pg.gen_policy() 