import os 
from LLMGuidedSeeding_pkg import * 
import requests

class ConversationalInterface:
    def __init__(
        self):
        self.this =  1 
        self.feedback = None 
        self.backend_url = 'http://127.0.0.1:5000/backend/process_message'

    def ask_human(self,content):
        '''
        Make a post request to the backend flask server with the content we want to display to the human
        '''
        print("Question: ",content)
        payload = {'message': content}
        response = requests.post(self.backend_url, json = payload)
        if response.ok:
            print('Feedback sent to backend successfully! \n')
        else:
            print(f'Error sending feedback: {response.text} \n' )
    

    def get_human_feedback(self):
        '''
        This function ports text from the interface to the robot
        '''
        '''
        with open("feedback.txt","r") as f:
            feedback = f.read()
        '''
        feedback = input("Feedback: ")
        self.feedback = feedback
        return feedback
    
        
    def ask_policy_verification(self,policy):
        '''
        This function enhances the policy for verification by the human 
        '''
        preface = "I believe this policy should complete the desired task. What do you think?"
        enhanced_prompt = preface + "\n" + policy 
        self.ask_human(enhanced_prompt)
        with open("prompts/verification_prompt.txt","r") as f:
            prompt = f.read()
        enhanced_verification_response = prompt + self.get_human_feedback()
        print("enhanced verification response: ",enhanced_verification_response)
        llm_result = generate_with_openai(enhanced_verification_response)
        print("verifying the policy; this is the llm result:",llm_result)
        if "true" in llm_result or "True" in llm_result:
            return True 
        elif "false" in llm_result or "False" in llm_result:
            return False 
        else:
            return False 
        
    def ask_object_clarification(self,obj):
        '''
        This function enhances the prompt to ask for a description of an unknown object
        '''
        with open("prompts/object_clarification_prompt.txt","r") as f:
            prompt = f.read()
        enhanced_prompt = prompt.replace("*INSERT_OBJECT*",obj)
        #self.ask_human(enhanced_prompt)