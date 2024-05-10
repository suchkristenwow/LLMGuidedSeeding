from LLMGuidedSeeding_pkg import generate_with_openai
from gen_utils import get_most_recent_file 

def code_gen(policy): 
    '''
    Given a numbered step policy, generate a python script that executes the policy 
    '''
    # Split the input text by newlines to separate each step
    steps = policy.strip().split('\n')
    # Remove any leading or trailing whitespace from each step
    steps = [step.strip() for step in steps]
    with open("/home/kristen/LLMGuidedSeeding/prompts/codeGen_prompt.txt","r") as f: 
        prompt = f.read()

    for step in steps: 
        enhanced_prompt = prompt.replace(step)

if __name__ == "__main__":
    policy_file = get_most_recent_file("/home/kristen/LLMGuidedSeeding/experiment_data","finalPolicy.txt",sub_dir_name="policy_generation_logs")
    with open(policy_file,"r") as f: 
        policy = f.read()
    code_gen(policy)
