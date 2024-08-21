from LLMGuidedSeeding_pkg.utils.llm_utils import generate_with_openai

prompt1 = "This is a test to see if the convsersation memory is working in my script. Remember this key word: cats" 
response, history = generate_with_openai(prompt1) 

prompt2 = "What is the key word? "
response, _ = generate_with_openai(prompt2,conversation_history=history)

print("response:",response) 