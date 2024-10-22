
    def gen_code_from_policy(self):
        print("reading the policy...")
        with open("policy.txt","r") as f:
            policy = f.read() 

        with open("prompts/step_parser_codeGen.txt","r") as f: 
            prompt = f.read() 

        enhanced_prompt = prompt.replace("*INSERT_QUERY*",self.query)
        enhanced_prompt = enhanced_prompt.replace("*INSERT_POLICY*",policy)

        raw_code,_ = generate_with_openai(enhanced_prompt)

        code = remove_chatGPT_commentary(raw_code)

        print("writing code to self_critique_logs/policy_execution.py")
        with open("self_critique_logs/policy_execution.py","w") as f:
            f.write(code) 