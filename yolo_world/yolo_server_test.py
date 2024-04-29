import requests
import cv2
import json

def send_request_to_server(image_path, caption, server_url, output_file):
    # Read the image and convert it to a format suitable for sending via HTTP
    image = cv2.imread(image_path)
    _, encoded_image = cv2.imencode('.jpg', image)
    image_bytes = encoded_image.tobytes()

    # Set the appropriate header for the request
    headers = {'content-type': 'image/jpeg'}

    # Send POST request to the server
    response = requests.post(server_url, data=image_bytes, headers=headers, params={'caption': caption})
    
    # Check the response and save it to a file
    if response.status_code == 200:
        response_data = response.json()
        with open(output_file, 'w') as file:
            json.dump(response_data, file, indent=4)
        return response_data
    else:
        error_message = f"Error: {response.status_code}"
        with open(output_file, 'w') as file:
            file.write(error_message)
        return error_message

# Specify the path to your image, the caption, and the output file
image_path = '../glip_server/test_2.jpg'  # Change this to the path of your image
#caption = ['person', 'belt']  # Change this to your desired caption
caption_list = ['couch', 'TV']
caption = ",".join(caption_list)
#caption = 'person'
server_url = 'http://localhost:5005/process'  # Change if your server is running on a different URL
output_file = 'output_file.json'  # Specify the path to the output file

# Send the request and get the response
response = send_request_to_server(image_path, caption, server_url, output_file)
print(json.dumps(response, indent=4))
