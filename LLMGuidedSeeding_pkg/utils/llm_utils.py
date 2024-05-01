import requests 
import os 
import logging 
from requests.exceptions import SSLError
import time 

def generate_with_openai(prompt, max_retries = 5, retry_delay = 10, n_predict=2048, temperature=0.9, top_p=0.9):
    attempts = 0 
    openai_api_key = os.getenv("openai_key")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful, respectful, and honest assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": n_predict,
        "temperature": temperature,
        "top_p": top_p,
    }
    while attempts < max_retries:
        try: 
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", json=data, headers=headers
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logging.error(f"Error generating text with OpenAI: {response.text}")
                return None
        except SSLError as e: 
            print(f"Encountered SSLError: {e}")
            attempts += 1
            if attempts < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Unable to complete request.")
                raise
