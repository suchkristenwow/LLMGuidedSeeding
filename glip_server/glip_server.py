from flask import Flask, request, jsonify
import numpy as np
import time
from io import BytesIO
from PIL import Image
import sys
import torch

# GLIP related imports, adjust the paths as per your setup
sys.path.append('GLIP')
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

class GLIPInference:
    def __init__(self, config_file, weight_file, device='cuda'):
        # Configure and initialize the model
        cfg.local_rank = 0
        cfg.num_gpus = 1
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        cfg.merge_from_list(["MODEL.DEVICE", device])

        self.glip_demo = GLIPDemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.8,
            show_mask_heatmaps=False
        )

    @staticmethod
    def load_image(image_data):
        """Load an image from binary data."""
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        return np.array(pil_image)[:, :, [2, 1, 0]]  # Convert to BGR format

    def infer_bounding_boxes(self, image, caption):
        """Perform inference and return bounding boxes."""
        bbox_list = self.glip_demo.inference(image, caption)
        scores = bbox_list.get_field("scores")
        labels = bbox_list.get_field("labels")
        bounding_boxes = bbox_list.bbox
        mode = bbox_list.mode

        if mode == "xyxy":
            x_coordinates = bounding_boxes[:, [0, 2]]  # x_min and x_max
            y_coordinates = bounding_boxes[:, [1, 3]]  # y_min and y_max
        elif mode == "xywh":
            x_coordinates = bounding_boxes[:, 0]  # x
            y_coordinates = bounding_boxes[:, 1]  # y
        else:
            raise ValueError(f"Unsupported bounding box mode: {mode}")

        return x_coordinates, y_coordinates, scores, labels, mode

# Initialize Flask app
app = Flask(__name__)

# Initialize GLIP model, adjust paths as needed
config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml"
weight_file = "GLIP/MODEL/glip_large_model.pth"
glip = GLIPInference(config_file, weight_file)

@app.route('/process', methods=['POST'])
def process_image():
    start_time = time.time()

    # Receive the image
    encoded_image = request.data
    image = GLIPInference.load_image(encoded_image)

    # Get caption from request
    caption = request.args.get('caption', '')

    # Process the image with GLIP
    x_coords, y_coords, scores, labels, bbox_mode = glip.infer_bounding_boxes(image, caption)

    # Prepare the base response data
    response_data = {
        'caption': caption,
        'processing_time': time.time() - start_time
    }

    # Check if scores tensor is empty
    if scores.numel() == 0:
        response_data.update({
            'x_coords': [],
            'y_coords': [],
            'scores': [],
            'labels': []
        })
    else:
        # Check if there are three or fewer detections
        if scores.size(0) <= 3:
            # Skip quantile filtering, return all detections
            labels_filtered_strings = [glip.glip_demo.entities[index-1] for index in labels]
            response_data.update({
                'x_coords': x_coords.tolist(),
                'y_coords': y_coords.tolist(),
                'scores': scores.tolist(),
                'labels': labels_filtered_strings,
                'bbox_mode': bbox_mode
            })
        else:
            # Proceed with quantile filtering for more than three detections
            min_score = scores.min()
            max_score = scores.max()
            scores_normalized = (scores - min_score) / (max_score - min_score)

            top_quartile_threshold = torch.quantile(scores_normalized, 0.75)
            top_quartile_indices = scores_normalized >= top_quartile_threshold

            x_coords_filtered = x_coords[top_quartile_indices]
            y_coords_filtered = y_coords[top_quartile_indices]
            labels_filtered = labels[top_quartile_indices]
            scores_filtered = scores[top_quartile_indices]
            labels_filtered_strings = [glip.glip_demo.entities[index-1] for index in labels_filtered]

            response_data.update({
                'x_coords': x_coords_filtered.tolist(),
                'y_coords': y_coords_filtered.tolist(),
                'scores': scores_filtered.tolist(),
                'labels': labels_filtered_strings,
                'bbox_mode': bbox_mode
            })

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
