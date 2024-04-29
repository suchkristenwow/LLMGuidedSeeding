import requests 
import os 
import logging 

def generate_with_openai(prompt, n_predict=2048, temperature=0.9, top_p=0.9):
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
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", json=data, headers=headers
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        logging.error(f"Error generating text with OpenAI: {response.text}")
        return None