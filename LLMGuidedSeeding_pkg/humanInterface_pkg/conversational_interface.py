import json
from LLMGuidedSeeding_pkg import * 

class ConversationalInterface:
    def __init__(self, robot_server):
        self.robot_api = robot_server
        self.conversation_states = []
        self.conversation_memory = 5
        self.problem_statement = ""
        self.context = ""
        self.conversation_active = False
        self.provision = None
        self.expiration = None

    def kickback_query(self):
        # Gather description of the situation
        situation_description = self.robot_api.get_situation_description()
        # Encode the result into a singular prompt
        #prompt = f"The robot is currently in the following situation: {situation_description}. What would you like to know more about?"
        return prompt

    def pose_question(self, question):
        # Send the question to the user via the Flask server
        messages.append(question)
        response = None
        while response is None:
            if len(messages) > len(question):
                response = messages[-1]
        return response

    def process_response(self, response):
        # Process the user's response and determine if a resolution has been reached
        resolved = False
        resolution_prompt = self.read_file_contents("prompts/tot_nl_generate.txt")
        resolution_prompt = resolution_prompt.replace("*INSERT_K_STATES*", str(self.conversation_states[:self.conversation_memory]))
        resolution_prompt = resolution_prompt.replace("*INSERT_PROBLEM_STATEMENT*", self.problem_statement)
        resolution_prompt = resolution_prompt.replace("*INSERT_RESPONSE*", response)
        #resolution_topic = TextGenerator.generate_text_with_hosted_model(resolution_prompt)
        resolution_topic = generate_with_openai(resolution_prompt)
        if "not" in resolution_topic:
            return resolved
        else:
            resolved = True
        return resolved

    def process_unresolved_response(self):
        self.context = ""
        context_addition_prompt = self.read_file_contents("prompts/add_context.txt")
        #context_addition_code = TextGenerator.generate_text_with_hosted_model(context_addition_prompt)
        context_addition_code = generate_with_openai(context_addition_prompt)
        exec(context_addition_code)
        context_integration_prompt = self.read_file_contents("prompts/integrate_context.txt")
        #context_addition_integrated = TextGenerator.generate_text_with_hosted_model(context_integration_prompt)
        context_addition_integrated = generate_with_openai(context_addition_integration_prompt)
        self.context = context_addition_integrated

    def run_conversation(self):
        # Kickback query to gather situation description
        self.problem_statement = self.kickback_query()
        # Set conversation_active to True
        self.conversation_active = True
        # Start the conversation loop
        while self.conversation_active:
            # Pose the current question to the user
            question = self.problem_statement
            response = self.pose_question(question)
            # Process the user's response
            resolved = self.process_response(response)
            if resolved:
                # Generate provision and expiration
                self.provision = self.generate_provision()
                self.expiration = self.generate_expiration()
                # End the conversation
                self.conversation_active = False
            else:
                # Update the prompt for the next iteration
                self.process_unresolved_response()

    def read_file_contents(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()