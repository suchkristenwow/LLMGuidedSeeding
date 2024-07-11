with open("/home/kristen/LLMGuidedSeeding/prompts/ex_policy.txt","r") as f:
    init_policy = f.read()

with open("/home/kristen/LLMGuidedSeeding/prompts/ex_feedback.txt","r") as f:
    user_feedback = f.read()

with open("/home/kristen/LLMGuidedSeeding/prompts/modify_policy.txt","r") as f:
    modify_prompt = f.read()

modify_prompt.replace("*INSERT_POLICY*",init_policy)
modify_prompt.replace("*INSERT_FEEDBACK*",user_feedback) 
