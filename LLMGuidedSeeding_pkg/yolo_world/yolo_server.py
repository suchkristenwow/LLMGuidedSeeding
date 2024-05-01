from flask import Flask, request, jsonify
import numpy as np
import time
from io import BytesIO
from PIL import Image
import sys
from ultralytics import YOLO
import torch
import clip
import argparse

args = None

class YoloWorldInference:
    def __init__(self, device="cuda"):
        # Configure and initialize the model
        self.yolo_model = YOLO("yolov8l-world.pt").to(device)
        self.prev_classes = ""
        self.clip_model, _ = clip.load("ViT-B/32")

    @staticmethod
    def load_image(image_data):
        """Load an image from binary data."""
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        return pil_image

    def infer(self, image, classes):
        """Perform inference and return bounding boxes."""
        if self.prev_classes != classes:
             self.set_classes(classes)
        results = self.yolo_model.predict(image, stream=True, max_det=args.max_det, conf=args.conf)
        return results

    # Note the built in set classes method doesn't keep clip in vram
    # Here we keep the model loaded to speed up inference time
    def set_classes(self, classes):
        device = next(self.clip_model.parameters()).device
        text_token = clip.tokenize(classes).to(device)
        txt_feats = self.clip_model.encode_text(text_token).to(dtype=torch.float32)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.yolo_model.model.txt_feats = txt_feats.reshape(
            -1, len(classes), txt_feats.shape[-1]
        )
        self.yolo_model.model.names = classes
        background = " "
        if background in classes:
            classes.remove(background)
            
        yolo_model_ref = self.yolo_model.model.model
        yolo_model_ref[-1].nc = len(classes)
        if self.yolo_model.predictor:
            self.yolo_model.predictor.model.names = classes


# Initialize Flask app
app = Flask(__name__)


# Initialize GLIP model, adjust paths as needed
yolo_world = YoloWorldInference()


@app.route("/process", methods=["POST"])
def process_image():
    start_time = time.time()

    # Receive the image
    encoded_image = request.data
    image = yolo_world.load_image(encoded_image)

    # Get caption from request
    caption = request.args.get("caption", "")
    caption = caption.split(",")

    boxes = []
    scores = []
    names = []

    # Process the image with Yolo World
    results = yolo_world.infer(image, caption)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        boxes_xyxy = boxes.xyxy
        x_coordinates = boxes_xyxy[:, [0, 2]]  # x_min and x_max
        y_coordinates = boxes_xyxy[:, [1, 3]]  # y_min and y_max
        scores = boxes.conf
        names = [result.names[int(idx)] for idx in boxes.cls.tolist()]


    # Prepare the base response data
    processing_time = time.time() - start_time
    response_data = {"caption": caption, "processing_time": processing_time}

    response_data.update(
        {
            "x_coords": x_coordinates.tolist(),
            "y_coords": y_coordinates.tolist(),
            "scores": scores.tolist(),
            "labels": names,
            "bbox_mode": "xyxy",
        }
    )
    
    print(f"Processed image with {len(names)} objects: {', '.join(names)} in {processing_time:.2f} seconds.")
    return jsonify(response_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yolo Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=5005, help="Port number")
    parser.add_argument("--max_det", type=int, default=10, help="Max detections")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
