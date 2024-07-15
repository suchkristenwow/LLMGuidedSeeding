import requests 
import os 
import logging 
from requests.exceptions import SSLError, ConnectionError 
import time 
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def generate_with_openai(prompt, max_retries=25, retry_delay=10, n_predict=2048, temperature=0.9, top_p=0.9, image_path=None):
    print("Tying to get an answer from Chat GPT hold on ...")
    attempts = 0
    openai_api_key = os.getenv("openai_key")

    if image_path is None:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        data = {
            "model": "gpt-4",
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
    else:
        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        data = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What’s in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

    response = None

    while attempts < max_retries:
        print("Trying to get response ...")
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", json=data, headers=headers
            )
            #print("response.status_code: ",response.status_code)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logging.error(f"Error generating text with OpenAI: {response.text}")
                return None
        except (SSLError, ConnectionError) as e:
            print(f"Encountered error: {e}")
            attempts += 1
            if attempts < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Unable to complete request.")
                raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise