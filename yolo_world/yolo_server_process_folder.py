import os
import glob
import cv2
import requests
import json
import argparse
import hashlib


def send_request_to_server(image_path, caption, server_url):
    image = cv2.imread(image_path)
    _, encoded_image = cv2.imencode(".jpg", image)
    image_bytes = encoded_image.tobytes()
    headers = {"content-type": "image/jpeg"}
    response = requests.post(
        server_url, data=image_bytes, headers=headers, params={"caption": caption}
    )
    return response




def get_color(label):
    """Generate a color for a given label using its hash value, biased towards red."""
    # Hash the label to ensure consistency across runs
    hash_val = int(hashlib.sha1(label.encode('utf-8')).hexdigest(), 16) % (2**24)
    # Bias towards red by adjusting the red component to be higher on average
    red = (hash_val & 255) | 128  # Ensures the red component is at least 128, making it more prominent
    green = (hash_val >> 8) & 255
    blue = (hash_val >> 16) & 255
    return (blue, green, red)


def save_output_image(image_path, response_data):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]  # Get the height and width of the image
    print("Found", len(response_data["labels"]), "objects")
    for i in range(len(response_data["labels"])):
        label = response_data["labels"][i]
        score = response_data["scores"][i]
        left, right = response_data["x_coords"][i]
        upper, lower = response_data["y_coords"][i]
        proj_x, proj_y = response_data["projection_points"][i]

        # Correct the assignment for proj_x and proj_y
        proj_x, proj_y = int(proj_x), int(proj_y)

        # Convert all coordinates to integers
        left, upper, right, lower = int(left), int(upper), int(right), int(lower)

        # Ensure coordinates are within the image boundaries
        left, right = max(0, left), min(width, right)
        upper, lower = max(0, upper), min(height, lower)

        # Get a unique color for each label
        color = get_color(label)

        # Draw the bounding box
        cv2.rectangle(image, (left, upper), (right, lower), color, 2)

        # Draw the projection point
        cv2.circle(image, (proj_x, proj_y), 5, (0, 0 ,255), -1)

        # Calculate the position for the label text to ensure it's always within the image
        text_x = max(0, left - 10)  # Shift text slightly to the left of the bounding box for visibility
        text_y = max(20, upper - 10)  # Position text above the bounding box; ensure it's not too close to the top edge
        if text_y < 20:  # If bounding box is too close to the top, put text inside the box at the top
            text_y = upper + 20

        # Put the label and score on the image
        cv2.putText(
            image,
            f"{label}: {score:.2f}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # Save the modified image to the specified output directory
    output_folder = os.path.join(os.path.dirname(image_path), "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)




parser = argparse.ArgumentParser(
    description="Send an image to the server for processing."
)
parser.add_argument(
    "--folder_path",
    type=str,
    required=True,
    help="The path to the folder containing image folders (left, right, front)",
)
parser.add_argument(
    "--caption",
    type=str,
    required=True,
    help="The caption to send to the server (comma-separated)",
)
parser.add_argument(
    "--server_url",
    type=str,
    default="http://localhost:5005/process",
    help="The URL of the server to which the image will be sent",
)
args = parser.parse_args()

# Create an output directory within each image folder if it doesn't exist
for folder in ["left", "right", "front"]:
    output_dir = os.path.join(args.folder_path, folder, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Process images in each folder
total_images = sum(len(glob.glob(os.path.join(args.folder_path, folder, "*.jpg"))) for folder in ["left", "right", "front"])
images_processed = 0
for folder in ["front", "right", "left"]:
    image_folder_path = os.path.join(args.folder_path, folder)
    for image_path in glob.glob(os.path.join(image_folder_path, "*.jpg")):
        images_processed += 1
        print(f"Processing {image_path}. Progress: {images_processed}/{total_images} images processed.")
        response = send_request_to_server(image_path, args.caption, args.server_url)
        if response.status_code == 200:
            response_data = response.json()
            save_output_image(image_path, response_data)
        else:
            print(f"Error processing {image_path}: {response.status_code}")
