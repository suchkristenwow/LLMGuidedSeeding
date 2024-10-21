import os 
from LLMGuidedSeeding_pkg import * 
import socketio
import time

class ConversationalInterface:
    def __init__(
        self):
        self.this =  1 
        self.feedback = None 
        self.human_response = False
        self.sio = socketio.Client()
        
        @self.sio.on('outgoing')
        def on_outgoing(data):
            self.feedback = data
            self.human_response = True

        self.sio.connect('http://localhost:7000')
        

    def ask_human(self,content):
        '''
        Emit the GPT content to the bakend through a socket 
        '''
        print("asking the hooman")
        # Send a message to the server 
        self.sio.emit('message', content)   


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
        self.human_response = False
        preface = "I believe this policy should complete the desired task. What do you think?"
        enhanced_prompt = preface + "\n" + policy 
        self.ask_human(enhanced_prompt)
        # wait for the user response. 
        while not self.human_response:
            time.sleep(1)
            
        with open("prompts/verification_prompt.txt","r") as f:
            prompt = f.read()
        enhanced_verification_response = prompt + self.feedback  #self.get_human_feedback()
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