from ultralytics import YOLO
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
from transformers import BertModel, BertTokenizer
import torch
from scipy import ndimage
from PIL import ImageDraw
import numpy as np
from scipy.spatial.distance import cosine
import cv2 as cv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm



class CosineSimilarityCalculator:
    def __init__(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()  # Set model to evaluation mode

    def get_embedding(self, text):
        # Encode text to get token ids and attention masks
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # Get embeddings from the BERT model
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the pooled output which is the aggregate representation for the sentence
        embeddings = outputs.pooler_output
        return embeddings

    def calculate_max_similarity(self, class1, class_list):
        embedding1 = self.get_embedding(class1).squeeze()
        max_similarity = -1
        max_class = None
        max_class_idx = -1
        
        for idx, class_instance in enumerate(class_list):
            embedding2 = self.get_embedding(class_instance).squeeze()
            # Calculate cosine similarity
            similarity = 1 - cosine(embedding1, embedding2)
            if similarity > max_similarity:
                max_similarity = similarity
                max_class = class_instance
                max_class_idx = idx
        
        return max_class_idx, max_class, max_similarity


class ObjectDetectorTest:
    def __init__(self, image) -> None:
        self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-coco")
        self.segmentation_model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")
        self.yolo_model= YOLO('yolov8l-world.pt')  # or choose yolov8m/l-world.pt
        self.image = image
        self.cosine_similarity_calculator = CosineSimilarityCalculator()
        
    def set_classes(self, classes):
        self.classes = classes
        self.yolo_model.set_classes(classes)
        
    def run_yolo(self, image_path):
        return self.yolo_model.predict(image_path)

    def get_segmentations_from_yolo(self, results):
        for result in results:
            boxes_xyxy = result.boxes.xyxy
            for idx, box in enumerate(boxes_xyxy):
                box_list = box.tolist()
                # Crop the image
                crop = self.image.crop((box_list[0], box_list[1], box_list[2], box_list[3]))
                label = [result.names[int(result.boxes[idx].cls)]]
                
                # Run segmentation on the crop
                segmentation_result = self.run_segmentation(crop)
                centroid = self.extract_labels_and_masks(segmentation_result, label)
                
                # Draw the centroid on the crop if one was found
                if centroid:
                    draw = ImageDraw.Draw(crop)
                    # Radius for the circle to be drawn
                    radius = 5
                    draw.ellipse([(centroid[0] - radius, centroid[1] - radius), (centroid[0] + radius, centroid[1] + radius)], fill='red', outline='red')
                
                # Display the crop with the centroid marked
                crop.show()

                
    def run_segmentation(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.segmentation_model(**inputs)
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        result = self.feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        return result
        
    def extract_labels_and_masks(self, result, label):
        segments_info = result['segments_info']
        segmentation_map = result['segmentation'].numpy()  # Convert segmentation tensor to numpy array

        # Initialization for the case of no match
        centroid = None

        # Finding the most similar segment
        print("Info: ", segments_info)
        segment_labels = [self.segmentation_model.config.id2label[segment['label_id']] for segment in segments_info]
        segment_labels = [item.replace('-merged', '') for item in segment_labels]
        print("Possible Lables: ", segment_labels)
        seg_idx, segment_name, similarity = self.cosine_similarity_calculator.calculate_max_similarity(label[0], segment_labels)
        self.draw_panoptic_segmentation(**result)
        print("Choosen Segment: ", segment_name, similarity)
        
        if seg_idx != -1:  # A similar segment was found
            segment_id_of_interest = segments_info[seg_idx]['id']
            binary_mask = segmentation_map == segment_id_of_interest

            labeled_array, num_features = ndimage.label(binary_mask)
            if num_features > 0:
                sizes = ndimage.sum(binary_mask, labeled_array, range(1, num_features + 1))
                largest_label = np.argmax(sizes) + 1
                largest_area_mask = labeled_array == largest_label

                centroid = ndimage.center_of_mass(largest_area_mask)
                # Note: centroid is in (row, col) format, might need (x, y) i.e., (col, row)
                centroid = (centroid[1], centroid[0])  # Converting to (x, y) format

        return centroid

            
    def process_masks(self, masks_queries_logits):
        # Assuming a sigmoid threshold to convert masks to binary
        threshold = 0.5
        masks = torch.sigmoid(masks_queries_logits) > threshold
        masks_np = masks.cpu().numpy()

        # Assuming we're working with the first mask for simplicity
        # In practice, you'd select the mask corresponding to the highest similarity segment
        first_mask = masks_np[0, 0, :, :]  # Adjust indexing based on your actual data structure

        # Find the largest connected component
        labeled_array, num_features = ndimage.label(first_mask)
        sizes = ndimage.sum(first_mask, labeled_array, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        largest_area_mask = labeled_array == largest_label

        # Calculate the centroid of the largest area
        centroid = ndimage.center_of_mass(largest_area_mask)
        return centroid
    
    
    def draw_panoptic_segmentation(self, segmentation, segments_info):
        # get the used color map
        viridis = cm.get_cmap('viridis', torch.max(segmentation))
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        instances_counter = defaultdict(int)
        handles = []
        # for each segment, draw its legend
        for segment in segments_info:
            segment_id = segment['id']
            print(segment)
            segment_label_id = segment['label_id']
            segment_label = self.segmentation_model.config.id2label[segment_label_id]
            label = f"{segment_label}-{instances_counter[segment_label_id]}"
            instances_counter[segment_label_id] += 1
            color = viridis(segment_id)
            handles.append(mpatches.Patch(color=color, label=label))
            
        ax.legend(handles=handles)
        plt.show()


    def run(self):
        results = self.run_yolo(self.image)
        self.get_segmentations_from_yolo(results)
        return results

if __name__ == "__main__":
    image_file = '../glip_server/test.jpg'
    image_file = '/home/harel/Pictures/Screenshots/test_image.jpg'
    object_detector = ObjectDetectorTest(Image.open(image_file))
    classes = ["desk", "keyboard", "plant", "refrigerator"]
    object_detector.set_classes(classes)
    results = object_detector.run()
    results[0].show()